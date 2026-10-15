#!/usr/bin/env python3
# SPDX-License-Identifier: MulanPSL-2.0
"""llm_detect_rbnx — object detection service with two resident backends.

Detects a named object in the camera frame and returns its 2D bbox. Two
detector backends are kept warm and selected PER CALL by the request's
``backend`` field:

  - ``yolo`` / ``yolo_http`` — host-side GPU YOLO server (:8770), fine-tuned
    on carrot/potato/tomato. Also returns an oriented box: bbox_2d gets a
    5th element carrying the OBB rotation.
  - ``llm`` / ``vlm`` — an OpenAI-compatible vision model (e.g. Gemini,
    Qwen-VL, GPT-4o): open-vocabulary, any object described in words.
    Drop-in replacement for the older yolo_world_rbnx path.
  - empty → the service's configured default (``detect_backend``).

Owns ``robonix/service/perception/object_detect/*``.

Exposes one atlas-routed MCP surface:

    robonix/service/perception/object_detect/detect_object

It returns the highest-confidence match for the requested name, with
2D bbox + 3D camera-frame centroid when depth is enabled (median depth
in bbox back-projected through the camera_info K matrix).

Lifecycle:
    on_init      — parse config (LLM endpoint/model/prompts + YOLO server
                   URL), bring up whichever backends are configured (only the
                   DEFAULT backend's absence is fatal), probe the YOLO server
                   best-effort, resolve atlas camera contracts, spawn the
                   rclpy background thread.
    on_deactivate — stop rclpy thread.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import toml
from openai import OpenAI

from robonix_api import ATLAS, Service, Ok, Err  # noqa: E402

# ── logging setup ───────────────────────────────────────────────────────────
# Use the same pattern as mid360_lidar_rbnx: simple basicConfig for stderr,
# then manually add a FileHandler. Use force=True (Python 3.8+) to ensure
# configuration takes effect even if robonix_api already configured the
# root logger during import.
_LOG_DIR = Path(os.environ.get(
    "LLM_DETECT_LOG_DIR",
    os.path.join(Path.home(), ".llm_detect", "logs"),
))
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = _LOG_DIR / "llm_detect.log"

# Directory for saving annotated detection images.
_DETECTION_IMG_DIR = Path(os.environ.get(
    "LLM_DETECT_IMG_DIR",
    os.path.join(Path.home(), ".llm_detect", "detections"),
))
_DETECTION_IMG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.environ.get("LLM_DETECT_LOG_LEVEL", "INFO"),
    format="[llm_detect] %(message)s",
    force=True,
)
log = logging.getLogger("llm_detect")
# Add file handler explicitly (append mode).
_file_handler = logging.FileHandler(str(_LOG_FILE), mode="a")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "[llm_detect] %(asctime)s %(levelname)s %(message)s"
))
log.addHandler(_file_handler)
log.info("log file: %s", _LOG_FILE)

# Provider id MUST match the deploy manifest's `service: - name: ...`.
llm_detect = Service(
    id=os.environ.get("ROBONIX_CAPABILITY_ID", "llm_detect"),
    namespace="robonix/service/perception/object_detect",
)

# ── shared state (between rclpy thread + MCP handlers) ──────────────────────
_state_lock = threading.Lock()
_initialized = False

# Cached cfg dict from on_init.
_LATEST_CFG: dict[str, Any] = {}

# Latest synchronized camera frame, captured by message_filters callback.
_latest_color_image = None       # numpy ndarray, BGR (from cv_bridge)
_latest_depth_image = None       # numpy ndarray, depth in mm (uint16)
_latest_camera_info = None       # sensor_msgs/CameraInfo

# Vertical-grasp mode: when true, depth is not consumed (object_center_3d
# is returned as []). This allows the depth subscriber to be skipped
# entirely, saving CPU + USB bandwidth. Set via cfg.skip_depth or env
# LLM_DETECT_SKIP_DEPTH=1.
_skip_depth = False

# LLM client (initialized in on_init).
_llm_client: Optional[OpenAI] = None
_llm_model: str = ""
_llm_temperature: float = 0.0
_prompts: dict[str, str] = {}
_rotation_cam2arm: bool = False

# ── detector-seam backends ──────────────────────────────────────────────────
# Both backends are resident at once; each detect_object call picks one:
#   "llm" / "vlm"  — OpenAI-compatible VLM, general/open-vocabulary detection.
#   "yolo_http" / "yolo" — host-side GPU YOLO server
#                 (docker/host-services/yolo_detect_server.py), reached over
#                 --net host on 127.0.0.1; fine-tuned on carrot/potato/tomato.
# The per-call `backend` field of DetectObject selects which one answers; if
# empty it falls back to cfg.detect_backend (_default_backend below). This lets
# Pilot route fruit -> YOLO and everything-else -> VLM by calling the matching
# skill, while both backends stay warm. Only _call_llm_detect /
# _detect_object_via_yolo_http differ; ROS + MCP + camera plumbing is shared.
_default_backend: str = "llm"     # cfg.detect_backend — used when a call omits `backend`
_llm_ready: bool = False          # VLM client + prompts loaded
_yolo_configured: bool = False         # yolo_http url 已配置（可达性只在 init 探测、非致命）
_yolo_http_url: str = "http://127.0.0.1:8770/detect"
_yolo_http_timeout_s: float = 10.0
_yolo_conf: Optional[float] = None   # None => the server's own default conf
_yolo_iou: Optional[float] = None    # None => the server's own default iou

# cv_bridge instance.
_bridge = None

# ROS thread state.
_ros_node = None
_ros_thread: Optional[threading.Thread] = None
_ros_stop_evt = threading.Event()

# Synchronization between init() and _ros_thread_main.
_ros_ready_evt = threading.Event()
_ros_thread_error: Optional[BaseException] = None
_ROS_READY_TIMEOUT_S = 15.0


# ── upstream-resolution helpers ─────────────────────────────────────────────
_DEFAULT_TOPICS = {
    "rgb":         "/camera/color/image_raw",
    "depth":       "/camera/depth/image_raw",
    "camera_info": "/camera/color/camera_info",
}

_DEP_CONTRACTS = {
    "rgb":         "robonix/primitive/camera/rgb",
    "depth":       "robonix/primitive/camera/depth",
    "camera_info": "robonix/primitive/camera/camera_info",
}


def _resolve_topic(key: str, cfg: dict) -> str:
    """Resolve the ROS topic name to subscribe for `key`.

    Priority:
      1. cfg[f'{key}_topic'] — explicit override
      2. atlas find_capability(<contract>, transport=ros2, provider_id=...)
         → endpoint
      3. _DEFAULT_TOPICS[key]
    """
    explicit = (cfg.get(f"{key}_topic") or "").strip()
    if explicit:
        log.info("topic[%s] explicit cfg override: %s", key, explicit)
        return explicit

    contract_id = _DEP_CONTRACTS[key]
    provider_id = str(cfg.get("camera_provider_id") or "").strip()
    try:
        caps = ATLAS.find_capability(
            contract_id=contract_id,
            transport="ros2",
            provider_id=provider_id,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("atlas query %s failed: %s — falling back to default",
                    contract_id, e)
        caps = []

    # Log every candidate atlas returned so multi-camera setups are debuggable.
    # `provider_id` is the only knob object_detect has for picking one — if you
    # see multiple lines here and the wrong one gets chosen, set
    # `camera_provider_id` in the deploy manifest to the desired provider.
    if caps:
        log.info(
            "topic[%s] atlas returned %d candidate(s) for %s "
            "(transport=ros2, provider_id_filter=%r):",
            key, len(caps), contract_id, provider_id or "<any>",
        )
        for i, c in enumerate(caps):
            try:
                kind_name = getattr(c.provider_kind, "name", str(c.provider_kind))
                transport_name = getattr(c.transport, "name", str(c.transport))
            except Exception:  # noqa: BLE001
                kind_name = str(getattr(c, "provider_kind", "?"))
                transport_name = str(getattr(c, "transport", "?"))
            log.info(
                "  [%d]%s provider_id=%s kind=%s transport=%s contract=%s "
                "ns_mismatch=%s desc=%r",
                i,
                " <- SELECTED" if i == 0 else "",
                c.provider_id,
                kind_name,
                transport_name,
                c.contract_id,
                getattr(c, "namespace_mismatch", False),
                getattr(c, "description", "") or "",
            )

        try:
            ch = llm_detect.connect_capability(caps[0], contract_id, "ros2")
            ep = ch.endpoint
            try:
                ch.close()
            except Exception:  # noqa: BLE001
                pass
            if ep:
                log.info("topic[%s] resolved via atlas: %s (provider=%s)",
                         key, ep, caps[0].provider_id)
                return ep
        except Exception as e:  # noqa: BLE001
            log.warning("atlas connect %s failed: %s", contract_id, e)

    fallback = _DEFAULT_TOPICS[key]
    if provider_id:
        log.warning(
            "topic[%s] no atlas provider %s for %s; using default %s",
            key, provider_id, contract_id, fallback,
        )
    else:
        log.warning("topic[%s] no atlas provider; using default %s", key, fallback)
    return fallback


# ── LLM detection helpers ───────────────────────────────────────────────────
def _extract_json_from_markdown(text: str) -> str:
    """Extract JSON content from possible markdown code fences."""
    m = re.search(r"```(?:json|JSON)\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, flags=re.S)
    if m:
        return m.group(1).strip()
    return text


def _call_llm_detect(image_bgr: np.ndarray, object_name: str) -> dict:
    """Call the LLM API with the image and object name, return parsed result.

    Returns dict with keys:
      - success:       bool, true if target object was found
      - message:       str, human-readable status
      - target:        dict | None, the matched target object (normalized coords)
      - other_objects: list[dict], always empty in single-target mode
    Each object dict has: class_name, box_center_x/y, box_width/height (0-1).
    """
    global _llm_client, _llm_model, _llm_temperature, _prompts, _rotation_cam2arm

    if _llm_client is None:
        log.error("_call_llm_detect: LLM client is None")
        return {"success": False, "message": "LLM client not initialized",
                "target": None, "other_objects": []}

    # Rotate 180 degrees if camera is mounted opposite to arm.
    import cv2
    img = image_bgr.copy()
    if _rotation_cam2arm:
        log.debug("rotating image 180 degrees (rotation_cam2arm=True)")
        img = cv2.rotate(img, cv2.ROTATE_180)

    img_h, img_w = img.shape[:2]
    log.info("_call_llm_detect: object='%s', image_size=%dx%d, model=%s",
             object_name, img_w, img_h, _llm_model)

    # Encode image to base64 JPEG.
    _, img_encoded = cv2.imencode(".jpg", img)
    image_base64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")
    log.debug("encoded image to base64 JPEG, length=%d chars", len(image_base64))

    # Build prompt from template.
    prompt_template = _prompts.get("single_detect_prompt", "")
    if not prompt_template:
        log.error("prompt template 'single_detect_prompt' not found in loaded prompts")
        return {"success": False, "message": "prompt template 'single_detect_prompt' not found",
                "target": None, "other_objects": []}
    prompt = prompt_template.replace("{object_name}", object_name)
    log.info("prompt (first 200 chars): %s", prompt[:200])

    detection_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "integer"},
            "class_name": {"type": "string"},
            "box_center_x": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "box_center_y": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "box_width": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "box_height": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "thinking_process": {
                "anyOf": [{"type": "string"}, {"type": "null"}]
            },
            "failed": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}]
            },
        },
        "required": [
            "id",
            "class_name",
            "box_center_x",
            "box_center_y",
            "box_width",
            "box_height",
            "thinking_process",
            "failed",
        ],
    }

    # Call LLM API.
    log.info("calling LLM API: base_url=%s, model=%s, temperature=%s",
             _llm_client.base_url, _llm_model, _llm_temperature)
    t_api = time.time()
    try:
        completion = _llm_client.chat.completions.create(
            model=_llm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            },
                        },
                    ],
                }
            ],
            temperature=_llm_temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "single_target_detection",
                    "strict": True,
                    "schema": detection_schema,
                },
            },
        )
    except Exception as e:  # noqa: BLE001
        elapsed_api = time.time() - t_api
        log.error("LLM API call FAILED after %.2fs: %s: %s",
                  elapsed_api, type(e).__name__, e)
        return {"success": False, "message": f"LLM API call failed: {e}",
                "target": None, "other_objects": []}

    elapsed_api = time.time() - t_api
    log.info("LLM API responded in %.2fs", elapsed_api)

    if not completion.choices:
        log.error("LLM returned no choices. completion=%s", completion)
        return {"success": False, "message": "LLM returned no choices",
                "target": None, "other_objects": []}

    content = completion.choices[0].message.content
    if not content:
        log.error("LLM returned empty content. finish_reason=%s",
                  completion.choices[0].finish_reason)
        return {"success": False, "message": "LLM returned empty content",
                "target": None, "other_objects": []}

    log.info("LLM raw response (first 500 chars): %s", content[:500])

    # Parse JSON response.
    try:
        raw_json = _extract_json_from_markdown(content)
        result = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError) as e:
        log.error("JSON parse failed: %s. Raw content: %s", e, content[:1000])
        return {"success": False, "message": f"Failed to parse LLM response: {e}",
                "target": None, "other_objects": []}

    log.info("parsed LLM result keys=%s, full=%s",
             list(result.keys()), result)

    if bool(result.get("failed", False)):
        return {
            "success": False,
            "message": f"object '{object_name}' not found by LLM",
            "target": None,
            "other_objects": [],
            "fuzzy_matched": False,
            "match_reason": "single-target failed=true",
        }

    try:
        cx = float(result["box_center_x"])
        cy = float(result["box_center_y"])
        w = float(result["box_width"])
        h = float(result["box_height"])
    except (KeyError, TypeError, ValueError) as e:
        log.warning("single-target response has invalid bbox fields: %s raw=%s",
                    e, result)
        return {
            "success": False,
            "message": f"LLM returned invalid bbox fields for '{object_name}': {e}",
            "target": None,
            "other_objects": [],
            "fuzzy_matched": False,
            "match_reason": "invalid bbox fields",
        }

    bad_values = [v for v in (cx, cy, w, h) if not (0.0 <= v <= 1.0)]
    if bad_values:
        msg = (
            f"LLM returned non-normalized bbox for '{object_name}': "
            f"cx={cx:.3f}, cy={cy:.3f}, w={w:.3f}, h={h:.3f}"
        )
        log.warning("%s", msg)
        return {
            "success": False,
            "message": msg,
            "target": None,
            "other_objects": [],
            "fuzzy_matched": False,
            "match_reason": "non-normalized bbox",
        }
    if w <= 1e-3 or h <= 1e-3:
        msg = (
            f"LLM returned degenerate bbox for '{object_name}': "
            f"w={w:.4f}, h={h:.4f}"
        )
        log.warning("%s", msg)
        return {
            "success": False,
            "message": msg,
            "target": None,
            "other_objects": [],
            "fuzzy_matched": False,
            "match_reason": "degenerate bbox",
        }

    if _rotation_cam2arm:
        cx = 1.0 - cx
        cy = 1.0 - cy

    target = {
        "class_name": str(result.get("class_name", object_name)),
        "box_center_x": cx,
        "box_center_y": cy,
        "box_width": w,
        "box_height": h,
        "target_match_score": 1.0,
    }

    log.info("target locked: class='%s' reason=single-target schema",
             target["class_name"])

    return {
        "success":        True,
        "message":        "",
        "target":         target,
        "other_objects":  [],
        "fuzzy_matched":  False,
        "match_reason":   "single-target schema",
    }


# ── yolo_http backend (host GPU YOLO server) ────────────────────────────────
def _probe_yolo_http_backend() -> None:
    """Best-effort GET /health at init. A mis-wired / down server surfaces as
    a WARNING, never an error: the server only needs to be up at detection
    time, so boot + activation must not depend on it."""
    import urllib.request
    health = _yolo_http_url.rsplit("/", 1)[0] + "/health"
    try:
        with urllib.request.urlopen(health, timeout=3.0) as resp:
            info = json.loads(resp.read().decode("utf-8"))
        log.info("yolo_http backend healthy: model=%s device=%s classes=%s",
                 info.get("model_path"), info.get("device"), info.get("classes"))
    except Exception as e:  # noqa: BLE001
        log.warning("yolo_http backend NOT reachable at %s (%s: %s) — start "
                    "docker/host-services/yolo_detect_server.py before "
                    "detection; boot continues.", health, type(e).__name__, e)


def _detect_object_via_yolo_http(color_img: np.ndarray, object_name: str) -> dict:
    """Detector-seam backend: forward the frame to the host GPU YOLO server
    and map its output to the seam shape. Server contract lives in
    docker/host-services/yolo_detect_server.py.

    The server returns, per detection, bbox_2d = [x_min, y_min, x_max, y_max,
    rotation_deg] — an axis-aligned box carrying the OBB's center + w/h with
    the OBB rotation as a 5th element. grasp_pose consumes that 5th element
    for its long-edge gripper-yaw rule, so bbox_2d is passed through verbatim
    (do NOT truncate to 4). Detections are already sorted by confidence and
    filtered by object_name server-side, so detections[0] is the best match.

    object_center_3d is always [] (vertical-grasp mode): grasp_pose derives
    the arm XY from the pixel bbox via its own 2D homography and ignores
    depth. No image rotation is applied — YOLO runs on the raw camera frame
    exactly as roboarm's proven pathway does, and roboarm's homography
    already encodes the camera<->arm geometry.
    """
    import cv2
    import urllib.request
    from urllib.parse import urlencode

    ok, enc = cv2.imencode(".jpg", color_img)
    if not ok:
        return {"success": False, "message": "could not JPEG-encode frame",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    params: dict[str, Any] = {}
    if object_name:
        params["object_name"] = object_name
    if _yolo_conf is not None:
        params["conf"] = _yolo_conf
    if _yolo_iou is not None:
        params["iou"] = _yolo_iou
    url = _yolo_http_url + (("?" + urlencode(params)) if params else "")

    t0 = time.time()
    try:
        req = urllib.request.Request(
            url, data=enc.tobytes(), method="POST",
            headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=_yolo_http_timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        log.error("yolo_http request to %s failed: %s: %s",
                  url, type(e).__name__, e)
        return {"success": False,
                "message": f"yolo_http backend unreachable: {e}",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}
    elapsed = time.time() - t0

    if not payload.get("success", False):
        return {"success": False,
                "message": f"yolo server error: {payload.get('message', '?')}",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    dets = payload.get("detections", [])
    if not dets:
        log.info("yolo_http: '%s' not detected (%.0fms, server %.0fms)",
                 object_name, elapsed * 1000.0,
                 float(payload.get("inference_ms", 0.0)))
        return {"success": False,
                "message": f"object '{object_name}' not detected by YOLO",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    top = dets[0]
    bbox = [float(v) for v in top.get("bbox_2d", [])]
    conf = float(top.get("confidence", 0.0))
    cls = str(top.get("object_name", object_name))
    log.info("yolo_http detected '%s' -> class='%s' conf=%.2f bbox=%s "
             "(%.0fms, server %.0fms)",
             object_name, cls, conf, bbox, elapsed * 1000.0,
             float(payload.get("inference_ms", 0.0)))
    return {
        "success": True,
        "message": f"detected '{object_name}' via YOLO ('{cls}', {elapsed:.2f}s)",
        "bbox_2d": bbox,
        "object_center_3d": [],
        "confidence": conf,
    }


# ── detection core ──────────────────────────────────────────────────────────
def _normalize_backend(name: Any) -> str:
    """Canonicalize a backend selector to 'llm' or 'yolo_http'.

    Accepts the skill-facing aliases ('vlm', 'yolo') and the internal names
    ('llm', 'yolo_http'). Empty/unknown -> '' (caller falls back to default).
    """
    s = str(name or "").strip().lower()
    if s in ("llm", "vlm"):
        return "llm"
    if s in ("yolo", "yolo_http"):
        return "yolo_http"
    return ""


def _detect_object(object_name: str, backend: str = "") -> dict:
    """Detection core. Returns a dict with the same keys both surfaces fill.

    Returns:
      {
        "success":          bool,
        "message":          str,
        "bbox_2d":          [x_min, y_min, x_max, y_max] (pixels) or [];
                            the yolo_http backend appends a 5th element
                            (OBB rotation in degrees) — do not assume 4,
        "object_center_3d": [x, y, z] (meters, camera optical) or [],
        "confidence":       float,
      }
    """
    global _latest_color_image, _latest_depth_image, _latest_camera_info

    with _state_lock:
        if _latest_color_image is None or _latest_camera_info is None:
            return {
                "success": False,
                "message": ("camera data not available "
                            "(waiting for RGB+camera_info)"),
                "bbox_2d": [],
                "object_center_3d": [],
                "confidence": 0.0,
            }
        color_img = _latest_color_image.copy()
        cam_info = _latest_camera_info
        depth_img = _latest_depth_image.copy() if _latest_depth_image is not None and not _skip_depth else None

    # Per-call backend routing. `backend` picks which resident detector
    # answers; empty falls back to the configured default. yolo_http forwards
    # the frame to the host GPU YOLO server; the LLM (VLM) path continues below.
    requested = _normalize_backend(backend)
    if backend and not requested:
        log.warning("unknown backend %r (expected 'yolo'/'vlm', or empty for "
                    "the default) — falling back to %r",
                    backend, _default_backend)
    chosen = requested or _default_backend
    log.info("_detect_object: object=%r backend_req=%r -> %s",
             object_name, backend, chosen)
    if chosen == "yolo_http":
        if not _yolo_configured:
            return {"success": False,
                    "message": "yolo backend requested but not configured "
                               "(no yolo_http_url)",
                    "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}
        return _detect_object_via_yolo_http(color_img, object_name)

    if not _llm_ready:
        return {"success": False,
                "message": "vlm backend requested but not configured "
                           "(no llm_base_url/llm_model)",
                "bbox_2d": [], "object_center_3d": [], "confidence": 0.0}

    # 1. LLM-based detection.
    img_h, img_w = color_img.shape[:2]
    log.info("_detect_object: object='%s', frame_size=%dx%d", object_name, img_w, img_h)
    t0 = time.time()
    llm_result = _call_llm_detect(color_img, object_name)
    elapsed = time.time() - t0
    log.info("LLM detection took %.2fs for '%s', success=%s",
             elapsed, object_name, llm_result.get("success", False))

    target = llm_result.get("target")
    other_objects = llm_result.get("other_objects", [])
    fuzzy_matched = bool(llm_result.get("fuzzy_matched", False))
    match_reason = llm_result.get("match_reason", "")
    if fuzzy_matched and target is not None:
        log.warning("[fuzzy] using FUZZY match for '%s' -> '%s' (score=%.2f); reason=%s",
                    object_name, target.get("class_name", "?"),
                    target.get("target_match_score", 0.0), match_reason)

    # Helper: convert a normalised obj dict to pixel bbox.
    def _obj_to_pixel_bbox(o):
        cx, cy = o["box_center_x"], o["box_center_y"]
        w,  h  = o["box_width"],    o["box_height"]
        return [
            int(max(0,     (cx - w / 2) * img_w)),
            int(max(0,     (cy - h / 2) * img_h)),
            int(min(img_w, (cx + w / 2) * img_w)),
            int(min(img_h, (cy + h / 2) * img_h)),
        ]

    if not llm_result.get("success", False):
        log.warning("detection failed for '%s': %s",
                    object_name, llm_result.get("message", "unknown"))
        # Save annotated image with all OTHER objects drawn (blue) so we can
        # at least see what's in the frame.
        _save_detection_image(color_img, object_name,
                              target_bbox=None,
                              other_objects=[
                                  {**o, "bbox": _obj_to_pixel_bbox(o)}
                                  for o in other_objects
                              ],
                              success=False)
        return {
            "success": False,
            "message": llm_result.get("message", "LLM detection failed"),
            "bbox_2d": [],
            "object_center_3d": [],
            "confidence": 0.0,
        }

    # 2. Convert target's normalized coords to pixel bbox.
    bbox = [float(v) for v in _obj_to_pixel_bbox(target)]
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    log.info("target pixel bbox: [%d, %d, %d, %d]", x_min, y_min, x_max, y_max)

    # 3. 3D back-projection from depth + intrinsics.
    #    Skipped in vertical mode (_skip_depth=True) — the downstream
    #    yolo_grasp_rbnx does its own ray-plane intersection using TF.
    if _skip_depth or depth_img is None:
        center_3d = None
        log.info("skip_depth: object_center_3d not computed (vertical mode)")
    else:
        center_3d = _back_project_3d(bbox, depth_img, cam_info)
    log.info("3D center: %s", center_3d)

    # Heuristic confidence based on bbox area ratio.
    bbox_area = (x_max - x_min) * (y_max - y_min)
    img_area = img_w * img_h
    confidence = min(1.0, max(0.5, 0.5 + 0.5 * bbox_area / img_area))
    log.info("confidence=%.3f (bbox_area=%d, img_area=%d)",
             confidence, bbox_area, img_area)

    # 4. Save annotated image with target (green/yellow) + other objects (blue).
    target_class = target.get("class_name", object_name)
    target_score = float(target.get("target_match_score", 0.0))
    if fuzzy_matched:
        target_label = (f"{target_class} [FUZZY {target_score:.2f}] "
                        f"req='{object_name}' ({confidence:.2f})")
    else:
        target_label = f"{target_class} ({confidence:.2f})"
    _save_detection_image(color_img, object_name,
                          target_bbox={
                              "bbox": [x_min, y_min, x_max, y_max],
                              "class_name": target_class,
                              "confidence": confidence,
                              "label": target_label,
                              "fuzzy": fuzzy_matched,
                          },
                          other_objects=[
                              {**o, "bbox": _obj_to_pixel_bbox(o)}
                              for o in other_objects
                          ],
                          success=True)

    if fuzzy_matched:
        msg = (f"detected '{object_name}' via LLM (FUZZY match -> "
               f"'{target_class}', score={target_score:.2f}, {elapsed:.2f}s)")
    else:
        msg = f"detected '{object_name}' via LLM ({elapsed:.2f}s)"

    return {
        "success":          True,
        "message":          msg,
        "bbox_2d":          bbox,
        "object_center_3d": center_3d if center_3d is not None else [],
        "confidence":       float(confidence),
    }


def _draw_one_box(img, bbox, label, color, thickness=2):
    """Draw one bbox + label on `img` in-place."""
    import cv2
    x_min, y_min, x_max, y_max = [int(v) for v in bbox]
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # Label background above the box (or below if no room).
        ly = max(y_min - 4, th + 4)
        cv2.rectangle(img, (x_min, ly - th - 4),
                      (x_min + tw + 4, ly), color, -1)
        cv2.putText(img, label, (x_min + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)


def _save_detection_image(image_bgr: np.ndarray, object_name: str,
                          target_bbox: "dict | None" = None,
                          other_objects: "list[dict] | None" = None,
                          success: bool = True):
    """Save an annotated detection image to disk for debugging.

    Always draws ALL detected objects so you can see what's in the frame:
      - target (strict success):  GREEN box, label "<class> (conf)"
      - target (FUZZY success):   YELLOW box, label "<class> [FUZZY <score>]"
      - target (failed):          none drawn (target not found)
      - other_objects:            BLUE boxes, label "<class> @<score>"

    Files use FIXED names (no timestamp) so each call overwrites:
      - latest_ok.jpg     when target was detected successfully
      - latest_fail.jpg   when target was NOT found
      - latest.jpg        always (most recent regardless of outcome)

    other_objects: list of dicts each containing keys:
      - "bbox":               [x_min, y_min, x_max, y_max] in pixels
      - "class_name":         string
      - "target_match_score": float (optional, drawn into label if present)
    target_bbox: dict with keys "bbox", "class_name", "confidence", and
      optional "label" (overrides default) and "fuzzy" (bool).
    """
    try:
        import cv2
        img = image_bgr.copy()
        other_objects = other_objects or []

        # 1. Draw all other objects in BLUE first (so target overlays them).
        for o in other_objects:
            bbox = o.get("bbox")
            if not bbox:
                continue
            cls = str(o.get("class_name", "?"))
            score = o.get("target_match_score")
            if score is not None:
                label = f"{cls} @{float(score):.2f}"
            else:
                label = cls
            _draw_one_box(img, bbox, label, color=(255, 128, 0), thickness=2)

        # 2. Draw target on top. Yellow if fuzzy, green if strict.
        if success and target_bbox is not None:
            is_fuzzy = bool(target_bbox.get("fuzzy", False))
            target_color = (0, 255, 255) if is_fuzzy else (0, 255, 0)  # BGR
            label = target_bbox.get("label") or (
                f"{target_bbox.get('class_name', object_name)} "
                f"({target_bbox.get('confidence', 0.0):.2f})"
            )
            _draw_one_box(img, target_bbox["bbox"], label,
                          color=target_color, thickness=3)
            # Overlay banner at top so the user immediately knows it was fuzzy.
            if is_fuzzy:
                cv2.putText(img, f"FUZZY MATCH: req='{object_name}'",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2, cv2.LINE_AA)
            filename = "latest_ok.jpg"
        else:
            # Target not found — annotate top of image with red FAILED text.
            msg = f"FAILED: '{object_name}' not found ({len(other_objects)} other objs)"
            cv2.putText(img, msg, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                        cv2.LINE_AA)
            filename = "latest_fail.jpg"

        filepath = _DETECTION_IMG_DIR / filename
        cv2.imwrite(str(filepath), img)
        # Also save a generic "latest" pointer that always reflects most recent.
        cv2.imwrite(str(_DETECTION_IMG_DIR / "latest.jpg"), img)
        is_fuzzy_log = (success and target_bbox is not None
                        and bool(target_bbox.get("fuzzy", False)))
        log.info("[save] annotated image saved: %s (target=%s%s, others=%d)",
                 filepath,
                 "OK" if success else "FAIL",
                 " FUZZY" if is_fuzzy_log else "",
                 len(other_objects))
    except Exception as e:  # noqa: BLE001
        log.warning("failed to save detection image: %s", e)


def _back_project_3d(bbox_2d, depth_img, cam_info):
    """Median-depth back-projection. Returns [x, y, z] meters, or None.

    Same algorithm as yolo_world_rbnx: median (not mean) on the bbox
    depth ROI to reject background/zero pixels. Depth in mm → meters.
    """
    try:
        x_min, y_min, x_max, y_max = [int(v) for v in bbox_2d]
        roi = depth_img[y_min:y_max, x_min:x_max]
        valid = roi[(roi > 0) & (roi < 3000)]   # mm, max 3m
        if len(valid) == 0:
            log.warning("back-project: no valid depth in bbox %s",
                        (x_min, y_min, x_max, y_max))
            return None
        z = float(np.median(valid)) / 1000.0    # mm → m

        cx_pix = (x_min + x_max) / 2.0
        cy_pix = (y_min + y_max) / 2.0
        K = cam_info.k
        fx, fy = float(K[0]), float(K[4])
        cx, cy = float(K[2]), float(K[5])

        return [
            (cx_pix - cx) * z / fx,
            (cy_pix - cy) * z / fy,
            z,
        ]
    except Exception as e:  # noqa: BLE001
        log.error("back-project failed: %s", e)
        return None


# ── ROS bring-up (background thread) ────────────────────────────────────────
def _ros_thread_main(rgb_topic: str, depth_topic: str, info_topic: str) -> None:
    """Subscribe to camera topics in one rclpy node."""
    global _ros_node, _bridge
    global _latest_color_image, _latest_depth_image, _latest_camera_info
    global _ros_thread_error

    node = None
    try:
        import rclpy                              # noqa: E402
        from rclpy.node import Node               # noqa: E402
        from sensor_msgs.msg import Image, CameraInfo  # noqa: E402
        from cv_bridge import CvBridge            # noqa: E402
        import message_filters                    # noqa: E402
        rclpy.init(args=None)
        _bridge = CvBridge()
        node = Node("llm_detect_node")
        _ros_node = node

        # Synchronized RGB + depth + camera_info subscribers.
        #
        # In vertical mode (_skip_depth=True), depth is NOT subscribed —
        # only RGB + camera_info are synchronized. This saves USB
        # bandwidth and decouples detection from depth availability.
        sub_rgb   = message_filters.Subscriber(node, Image,      rgb_topic)
        sub_info  = message_filters.Subscriber(node, CameraInfo, info_topic)

        if _skip_depth:
            # RGB + camera_info only (2-way sync).
            sync = message_filters.ApproximateTimeSynchronizer(
                [sub_rgb, sub_info], queue_size=10, slop=0.1)

            def _camera_cb(rgb_msg, info_msg):
                global _latest_color_image, _latest_depth_image, _latest_camera_info
                try:
                    rgb = _bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                except Exception as e:  # noqa: BLE001
                    node.get_logger().error(f"camera_cb cv_bridge: {e}")
                    return
                with _state_lock:
                    _latest_color_image = rgb
                    _latest_depth_image = None
                    _latest_camera_info = info_msg
            sync.registerCallback(_camera_cb)
            log.info("subscribed (skip_depth): rgb=%s  info=%s",
                     rgb_topic, info_topic)
        else:
            sub_depth = message_filters.Subscriber(node, Image, depth_topic)
            sync = message_filters.ApproximateTimeSynchronizer(
                [sub_rgb, sub_depth, sub_info], queue_size=10, slop=0.1)

            def _camera_cb(rgb_msg, depth_msg, info_msg):
                global _latest_color_image, _latest_depth_image, _latest_camera_info
                try:
                    rgb   = _bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding="bgr8")
                    depth = _bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
                except Exception as e:  # noqa: BLE001
                    node.get_logger().error(f"camera_cb cv_bridge: {e}")
                    return
                with _state_lock:
                    _latest_color_image = rgb
                    _latest_depth_image = depth
                    _latest_camera_info = info_msg
            sync.registerCallback(_camera_cb)
            log.info("subscribed: rgb=%s  depth=%s  info=%s",
                     rgb_topic, depth_topic, info_topic)

    except BaseException as e:  # noqa: BLE001
        _ros_thread_error = e
        log.error("rclpy thread setup failed: %s: %s",
                  type(e).__name__, e, exc_info=True)
        try:
            if node is not None:
                node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        try:
            import rclpy as _rclpy_for_shutdown
            if _rclpy_for_shutdown.ok():
                _rclpy_for_shutdown.shutdown()
        except Exception:  # noqa: BLE001
            pass
        _ros_ready_evt.set()
        return

    # Setup OK — let init() proceed.
    _ros_ready_evt.set()

    # Spin until told to stop.
    import rclpy  # noqa: E402
    while not _ros_stop_evt.is_set():
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except Exception as e:  # noqa: BLE001
            log.warning("rclpy.spin_once raised: %s", e)
    try:
        node.destroy_node()
    except Exception:  # noqa: BLE001
        pass
    try:
        rclpy.shutdown()
    except Exception:  # noqa: BLE001
        pass
    log.info("rclpy thread exited")


# ── lifecycle ───────────────────────────────────────────────────────────────
@llm_detect.on_init
def init(cfg):
    """Driver(CMD_INIT). Steps:
      1. parse cfg + initialize LLM client
      2. load prompts from config/prompts.toml
      3. resolve atlas camera contracts → topic names
      4. spawn rclpy thread (camera subscribers only)
    """
    global _initialized, _llm_client, _llm_model, _llm_temperature
    global _prompts, _rotation_cam2arm
    global _default_backend, _yolo_http_url, _yolo_http_timeout_s
    global _yolo_conf, _yolo_iou, _llm_ready, _yolo_configured
    with _state_lock:
        if _initialized:
            return Ok()

    cfg = cfg or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg) if cfg else {}
        except json.JSONDecodeError as e:
            return Err(f"bad config_json: {e}")

    global _LATEST_CFG
    _LATEST_CFG = cfg

    # 1. Initialize LLM client.
    llm_base_url = cfg.get("llm_base_url", "").strip()
    llm_api_key = cfg.get("llm_api_key", "any").strip()
    _llm_model = cfg.get("llm_model", "").strip()
    _llm_temperature = float(cfg.get("temperature", 0.0))
    _rotation_cam2arm = bool(cfg.get("rotation_cam2arm", False))

    # Default backend when a detect_object call omits `backend` (see module
    # notes). Normalize aliases: "vlm"->"llm", "yolo"->"yolo_http".
    _default_backend = _normalize_backend(cfg.get("detect_backend", "llm"))
    _yolo_http_url = str(cfg.get("yolo_http_url",
                                 "http://127.0.0.1:8770/detect")).strip()
    _yolo_http_timeout_s = float(cfg.get("yolo_http_timeout_s", 10.0))
    _yolo_conf = cfg.get("yolo_conf", None)
    _yolo_conf = float(_yolo_conf) if _yolo_conf is not None else None
    _yolo_iou = cfg.get("yolo_iou", None)
    _yolo_iou = float(_yolo_iou) if _yolo_iou is not None else None
    log.info("default_backend=%s", _default_backend)

    # Vertical-grasp mode: skip depth subscription + back-projection.
    # Enabled via cfg.skip_depth=true or env LLM_DETECT_SKIP_DEPTH=1.
    global _skip_depth
    _skip_depth = bool(cfg.get("skip_depth", False)) or \
        os.environ.get("LLM_DETECT_SKIP_DEPTH", "").lower() in ("1", "true", "yes")
    if _skip_depth:
        log.info("skip_depth ENABLED — depth not subscribed, "
                 "object_center_3d will be [] (vertical grasp mode)")
    # API request timeout in seconds. Default 30s — much shorter than
    # OpenAI SDK's default of 600s, so we don't hang indefinitely if the
    # upstream API stalls.
    llm_timeout = float(cfg.get("llm_timeout_s", 30.0))
    llm_max_retries = int(cfg.get("llm_max_retries", 1))

    # Bring up BOTH backends. Each is enabled when its config is present, so a
    # deploy can still run one alone (omit the VLM key, or omit yolo_http_url).
    # A backend is only fatal at init if it is the configured default; a
    # non-default backend that fails to configure just stays unavailable and
    # its route returns a clear error at call time.

    # --- VLM backend: OpenAI-compatible client + prompts ---
    if llm_base_url and _llm_model:
        log.info("initializing LLM client: base_url=%s model=%s temperature=%s timeout=%.1fs max_retries=%d",
                 llm_base_url, _llm_model, _llm_temperature, llm_timeout, llm_max_retries)
        try:
            _llm_client = OpenAI(
                base_url=llm_base_url,
                api_key=llm_api_key,
                timeout=llm_timeout,
                max_retries=llm_max_retries,
            )
        except Exception as e:  # noqa: BLE001
            return Err(f"Failed to create OpenAI client: {e}")

        pkg_root = Path(os.environ.get(
            "RBNX_PACKAGE_ROOT",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
        ))
        prompts_path = cfg.get("prompts_file") or str(pkg_root / "config" / "prompts.toml")
        if not Path(prompts_path).is_file():
            return Err(f"Prompts file not found: {prompts_path}")
        try:
            prompts_data = toml.load(prompts_path)
            _prompts = prompts_data.get("prompts", {})
        except Exception as e:  # noqa: BLE001
            return Err(f"Failed to load prompts: {e}")
        log.info("loaded %d prompt templates from %s", len(_prompts), prompts_path)
        _llm_ready = True
    else:
        log.info("VLM backend NOT configured (no llm_base_url/llm_model) — "
                 "'vlm' route will be unavailable")

    # --- YOLO backend: host GPU server. Probe now so a mis-wire surfaces
    # early (non-fatal — the server only has to be up at detection time). ---
    if _yolo_http_url:
        _probe_yolo_http_backend()
        _yolo_configured = True
    else:
        log.info("YOLO backend NOT configured (no yolo_http_url) — "
                 "'yolo' route will be unavailable")

    # The configured default backend must actually be available.
    if _default_backend == "llm" and not _llm_ready:
        return Err("detect_backend='llm' but VLM is not configured "
                   "(need llm_base_url + llm_model)")
    if _default_backend == "yolo_http" and not _yolo_configured:
        return Err("detect_backend='yolo_http' but yolo_http_url is empty")

    # 3. Resolve atlas camera contracts.
    rgb_topic   = _resolve_topic("rgb",         cfg)
    info_topic  = _resolve_topic("camera_info", cfg)
    if _skip_depth:
        depth_topic = _DEFAULT_TOPICS["depth"]
        log.info("skip_depth: depth topic resolution skipped")
    else:
        depth_topic = _resolve_topic("depth", cfg)

    # 4. Spawn rclpy thread.
    global _ros_thread, _ros_thread_error
    _ros_stop_evt.clear()
    _ros_ready_evt.clear()
    _ros_thread_error = None
    _ros_thread = threading.Thread(
        target=_ros_thread_main,
        args=(rgb_topic, depth_topic, info_topic),
        name="llm_detect-ros",
        daemon=True,
    )
    _ros_thread.start()

    if not _ros_ready_evt.wait(timeout=_ROS_READY_TIMEOUT_S):
        _ros_stop_evt.set()
        _ros_thread.join(timeout=2.0)
        return Err(
            f"rclpy thread did not become ready within "
            f"{_ROS_READY_TIMEOUT_S}s"
        )

    if _ros_thread_error is not None:
        err = _ros_thread_error
        _ros_stop_evt.set()
        _ros_thread.join(timeout=2.0)
        return Err(
            f"rclpy thread setup failed: {type(err).__name__}: {err}"
        )

    with _state_lock:
        _initialized = True
    log.info("init complete: object_detect MCP live")
    return Ok()


@llm_detect.on_deactivate
def deactivate():
    """ACTIVE → INACTIVE. Stop the rclpy thread."""
    log.info("CMD_DEACTIVATE: stopping rclpy thread")
    _ros_stop_evt.set()
    if _ros_thread is not None:
        _ros_thread.join(timeout=5.0)
    with _state_lock:
        global _initialized
        _initialized = False
    return Ok()


# ── atlas-routed MCP handler (Pilot's view) ─────────────────────────────────
from perception_mcp import (  # noqa: E402  pylint: disable=wrong-import-position
    DetectObject_Request, DetectObject_Response,
)


@llm_detect.mcp("robonix/service/perception/object_detect/detect_object")
def detect_object(req: DetectObject_Request) -> DetectObject_Response:
    """Locate a named object in the camera frame; return its 2D bbox.

    Two detector backends are resident; `backend` selects one per call:
      - "yolo": host GPU model fine-tuned on carrot / potato / tomato only —
        highest accuracy and an oriented box for those three fruits.
      - "vlm":  open-vocabulary vision model — any object described in words,
        no training needed (use for everything that is not those fruits).
      - "" (empty): use the service's configured default backend.
    Callers normally leave `backend` empty and let the pick/grab skill they
    invoked set it; this tool is not usually called directly by Pilot.
    """
    result = _detect_object(req.object_name, getattr(req, "backend", "") or "")
    return DetectObject_Response(
        bbox_2d          = list(result["bbox_2d"]),
        object_center_3d = list(result["object_center_3d"]),
        confidence       = float(result["confidence"]),
        success          = bool(result["success"]),
        message          = str(result["message"]),
    )


def main() -> int:
    import signal
    def _on_signal(sig, _frame):
        log.info("signal %d — shutting down", sig)
        _ros_stop_evt.set()
        if _ros_thread is not None:
            _ros_thread.join(timeout=3.0)
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT,  _on_signal)
    try:
        llm_detect.run()
    finally:
        _ros_stop_evt.set()
        if _ros_thread is not None:
            _ros_thread.join(timeout=3.0)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
