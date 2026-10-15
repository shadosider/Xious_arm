#!/usr/bin/env python3
"""Host-side YOLO object-detection HTTP server (GPU half of the detector seam).

WHY THIS EXISTS
---------------
The robonix grasp pipeline runs inside a ROS 2 container that deliberately has
no CUDA / torch / ultralytics (see docker/Dockerfile). Real YOLO inference must
therefore run on the HOST, in this service's own venv that yolo/setup_venv.sh
builds (ultralytics + torch; 脚本按板子挑 torch 轮子，见 torch-orin.txt /
torch-thor.txt). This script is that host-side
inference server: the pipeline's detector "seam"
(robonix/service/perception/object_detect/detect_object) is filled by a small
container-side shim that forwards camera frames here over HTTP (the container
shares the host network via --net host, so 127.0.0.1 is reachable).

Keeping inference behind a plain HTTP endpoint is what makes the seam swappable:
today it serves the trained YOLO OBB model; swapping YOLO -> a VLA model later is
just pointing the shim at a different server, with no pipeline changes.

WHAT IT REUSES (model + data pathway, NOT roboarm's code)
---------------------------------------------------------
Only the trained weights (now vendored at weights/best.pt) and the ultralytics
OBB inference call are reused. The pathway mirrors roboarm's
detect_objects_in_frame():
    results = model(frame, conf, iou)[0]
    results.obb.xywhr  -> (u, v, w, h, r)   # center px, size px, rotation rad
    results.obb.conf   -> score
    results.obb.cls    -> class id  -> model.names[id]

OUTPUT MAPPING TO THE SEAM
--------------------------
The pipeline's DetectedObject.bbox_2d is [x_min, y_min, x_max, y_max] (axis
aligned), and yolo_grasp accepts an optional 5th element = rotation in degrees.
To let yolo_grasp reconstruct the exact OBB (same center, w/h and rotation that
roboarm uses for the gripper-yaw long-edge rule), each detection is emitted as:
    bbox_2d = [u - w/2, v - h/2, u + w/2, v + h/2, rad2deg(r)]
so (x_min+x_max)/2 == u, (y_min+y_max)/2 == v, (x_max-x_min) == w,
(y_max-y_min) == h, and bbox_2d[4] == rotation_deg.

SECURITY
--------
No authentication. Bind to 127.0.0.1 only (default) so it is reachable from the
--net host container but NOT from the outside network. Do not expose on 0.0.0.0
on a shared/production host.

RUN (with this service's own venv, host-services/yolo/.venv)
------------------------------------------------------------
    # Build it once (venv 不在时 run 脚本也会自动建):
    #   bash host-services/yolo/setup_venv.sh
    # Then either use the wrapper, or call the server directly.
    # --model defaults to the vendored weights/best.pt next to this script,
    # so the repo is self-contained; pass --model to use a newer run.
    bash host-services/yolo/run_yolo_server.sh

    host-services/yolo/.venv/bin/python \
        host-services/yolo/yolo_detect_server.py \
        --host 127.0.0.1 --port 8770

    curl -s 127.0.0.1:8770/health | python3 -m json.tool
    curl -s --data-binary @frame.jpg 127.0.0.1:8770/detect | python3 -m json.tool
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

log = logging.getLogger("yolo_detect_server")

# Loaded once in main(); guarded by _MODEL_LOCK because ultralytics predict is
# not guaranteed re-entrant across threads.
_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_PATH = ""
_DEVICE = ""
_DEFAULT_CONF = 0.8
_DEFAULT_IOU = 0.45


def _class_names() -> dict:
    """model.names as an ordinary {int: str} dict (ultralytics returns a dict)."""
    names = getattr(_MODEL, "names", {}) or {}
    return {int(k): str(v) for k, v in names.items()}


def _run_detection(frame, conf: float, iou: float) -> list[dict]:
    """Reuse roboarm's OBB pathway; map each detection to the seam shape."""
    with _MODEL_LOCK:
        results = _MODEL(frame, conf=conf, iou=iou, verbose=False)[0]

    obb = getattr(results, "obb", None)
    if obb is None or obb.xywhr is None or len(obb.xywhr) == 0:
        return []

    xywhr = obb.xywhr.cpu().numpy()          # (N, 5): u, v, w, h, r(rad)
    scores = obb.conf.cpu().numpy()          # (N,)
    class_ids = obb.cls.cpu().numpy().astype(int)  # (N,)
    names = _class_names()

    out: list[dict] = []
    for (u, v, w, h, r), score, cid in zip(xywhr, scores, class_ids):
        u, v, w, h, r = float(u), float(v), float(w), float(h), float(r)
        out.append({
            "object_name": names.get(int(cid), str(int(cid))),
            "class_id": int(cid),
            "confidence": float(score),
            # OBB (center px, size px, rotation rad) — raw pathway output.
            "obb": [u, v, w, h, r],
            # Seam mapping: unrotated box at (u,v) with the OBB w/h + rotation
            # in DEGREES as the 5th element (yolo_grasp re-rotates it).
            "bbox_2d": [u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0,
                        math.degrees(r)],
        })
    out.sort(key=lambda d: d["confidence"], reverse=True)
    return out


class _Handler(BaseHTTPRequestHandler):
    server_version = "yolo_detect_server/1.0"

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # route access logs through logging
        log.debug("%s - %s", self.address_string(), fmt % args)

    def do_GET(self):
        if urlparse(self.path).path != "/health":
            self._send_json(404, {"success": False, "message": "not found"})
            return
        self._send_json(200, {
            "success": True,
            "status": "ok",
            "model_path": _MODEL_PATH,
            "device": _DEVICE or "auto",
            "classes": _class_names(),
            "default_conf": _DEFAULT_CONF,
            "default_iou": _DEFAULT_IOU,
        })

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/detect":
            self._send_json(404, {"success": False, "message": "not found"})
            return

        qs = parse_qs(parsed.query)
        try:
            conf = float(qs.get("conf", [_DEFAULT_CONF])[0])
            iou = float(qs.get("iou", [_DEFAULT_IOU])[0])
        except (TypeError, ValueError):
            self._send_json(400, {"success": False,
                                  "message": "conf/iou must be numeric"})
            return
        object_name = qs.get("object_name", [""])[0].strip()

        try:
            length = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            length = 0
        if length <= 0:
            self._send_json(400, {"success": False,
                                  "message": "empty body; POST raw image bytes"})
            return

        raw = self.rfile.read(length)
        try:
            with open("/tmp/yolo_server_last_raw.jpg", "wb") as _f:
                _f.write(raw)
            log.info("DEBUG_RAW %d bytes saved to /tmp/yolo_server_last_raw.jpg "
                     "(object_name=%r conf=%s iou=%s)",
                     len(raw), object_name, conf, iou)
        except Exception as _e:
            log.warning("DEBUG_RAW save failed: %s", _e)
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            log.warning("DEBUG_DECODE_FAIL len=%d first_bytes=%s",
                        len(raw), raw[:12].hex())
            self._send_json(400, {"success": False,
                                  "message": "could not decode image bytes"})
            return
        try:
            cv2.imwrite("/tmp/yolo_server_last_decoded.png", frame)
            log.info("DEBUG_DECODED shape=%s mean=%.1f",
                     frame.shape, float(frame.mean()))
        except Exception as _e:
            log.warning("DEBUG_DECODED save failed: %s", _e)

        t0 = time.time()
        try:
            detections = _run_detection(frame, conf, iou)
        except Exception as e:  # noqa: BLE001 — surface inference errors as 500
            log.exception("inference failed")
            self._send_json(500, {"success": False,
                                  "message": f"inference error: {e}"})
            return
        dt_ms = (time.time() - t0) * 1000.0
        log.info("DEBUG_RAW_DETS n=%d %s",
                 len(detections),
                 [(d["object_name"], round(d["confidence"], 3)) for d in detections])

        if object_name:
            key = object_name.lower()
            filtered = [d for d in detections
                        if key in d["object_name"].lower()]
        else:
            filtered = detections
        log.info("DEBUG_FILTERED object_name=%r kept=%d/%d",
                 object_name, len(filtered), len(detections))

        self._send_json(200, {
            "success": True,
            "message": f"{len(filtered)} detection(s)",
            "image_size": [int(frame.shape[0]), int(frame.shape[1])],
            "inference_ms": round(dt_ms, 1),
            "detections": filtered,
        })


def _warmup() -> None:
    """One inference on a black frame so the first real request isn't slow and
    any CUDA / weights problem surfaces at startup, not mid-pipeline."""
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        _run_detection(dummy, _DEFAULT_CONF, _DEFAULT_IOU)
        log.info("warmup inference ok")
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed")
        raise


def main() -> None:
    global _MODEL, _MODEL_PATH, _DEVICE, _DEFAULT_CONF, _DEFAULT_IOU

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Vendored weight (host-services/yolo/weights/best.pt) so the repo is
    # self-contained; override with --model to point at a newer training run.
    default_model = str(Path(__file__).resolve().parent / "weights" / "best.pt")
    ap.add_argument("--model", default=default_model)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address; keep 127.0.0.1 on shared hosts")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--conf", type=float, default=0.8)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default="",
                    help="'' = ultralytics auto (cuda if available), or e.g. cuda:0 / cpu")
    ap.add_argument("--log", default="info")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    _MODEL_PATH = args.model
    _DEVICE = args.device
    _DEFAULT_CONF = args.conf
    _DEFAULT_IOU = args.iou

    from ultralytics import YOLO  # imported here so --help works without torch
    log.info("loading model: %s", _MODEL_PATH)
    _MODEL = YOLO(_MODEL_PATH)
    if _DEVICE:
        _MODEL.to(_DEVICE)
    log.info("classes: %s", _class_names())
    _warmup()

    httpd = ThreadingHTTPServer((args.host, args.port), _Handler)
    log.info("serving on http://%s:%d  (POST /detect, GET /health)",
             args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
