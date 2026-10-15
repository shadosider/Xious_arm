# service-object-detect-rbnx

Robonix package for **object detection with two resident backends**: a fine-tuned host-side GPU YOLO server and an open-vocabulary VLM. It exposes an atlas-routed MCP service; the VLM backend replaces the older `yolo_world_rbnx` path. Owns the `service/perception/object_detect/*` namespace.

Catalog name: `robonix.service.object_detect`.

## Capability surface

| Contract                                                  | Mode | Transport | Source / handler                                             |
| --------------------------------------------------------- | ---- | --------- | ------------------------------------------------------------ |
| `robonix/service/perception/object_detect/driver`         | rpc  | gRPC      | `Driver(CMD_INIT, config_json)` — lifecycle gate             |
| `robonix/service/perception/object_detect/detect_object`  | rpc  | MCP       | `DetectObject(object_name, backend) → bbox_2d + object_center_3d` |

There is no legacy `/yolo/detect_object` ROS-service fallback in this package.

## Detector backends

Both backends are **resident at once**; the request's `backend` field picks one per call:

| `backend` | Backend | Knows | `bbox_2d` |
| --------- | ------- | ----- | --------- |
| `"yolo"` (alias `yolo_http`) | host-side GPU YOLO server — `docker/host-services/yolo_detect_server.py`, `127.0.0.1:8770` | only the three fine-tuned fruits (carrot / potato / tomato) | `[x_min, y_min, x_max, y_max, rotation_deg]` — **5 elements** (oriented box) |
| `"vlm"` (alias `llm`) | OpenAI-compatible vision model | anything described in words | `[x_min, y_min, x_max, y_max]` — 4 elements |
| `""` (empty) | the configured default (`detect_backend`) | — | — |

Routing lives in the **skills**, not in this service: `pick` / `grab` call with `backend="yolo"`, `pick_general` / `grab_general` with `backend="vlm"`. Init brings up whichever backends are configured — only the **default** backend's absence is fatal; a non-default backend that is missing or down simply makes its route unavailable, with a clear error message at call time.

## Compared with the old `yolo_world` path

- the **VLM backend** replaces the old open-vocabulary path: RGB → OpenAI-compatible vision model → 2D bbox, with no local model files;
- the **`yolo` backend** adds a host-side GPU YOLO server for the three fine-tuned fruits — better accuracy and an oriented box (5-element `bbox_2d`), at the cost of running `docker/host-services/yolo_detect_server.py` on the host;
- in vertical-grasp mode the depth stream is completely unused (`skip_depth=true`); `service-grasp-pose-rbnx` maps the bbox center through the calibrated 2D homography and combines it with the configured tabletop z.

## Boot ordering

Must come **after** `primitive-orbbec-dabai_dcw-camera-rbnx` in the deploy manifest — Init resolves the RGB topic (and, when `skip_depth=false`, the depth topic) at atlas-registration time.

## Driver-init lifecycle

`start.sh` brings up the atlas bridge — no ROS spawn. The bridge opens a gRPC server, registers the capability and declares only `object_detect/driver`, then blocks awaiting `Driver(CMD_INIT, config_json)`.

When `rbnx boot` invokes Init the handler:

1. validates the LLM endpoint / API key / model name;
2. resolves `primitive/camera/rgb` and `primitive/camera/camera_info` via atlas (depth is skipped when `skip_depth=true`);
3. declares `object_detect/detect_object` on atlas.

## Layout

```
service-object-detect-rbnx/
├── package_manifest.yaml
├── capabilities/
│   └── service/perception/object_detect/{driver,detect_object}.v1.toml
├── config/                     example .env / prompt templates
├── llm_detect/                 Python package: atlas bridge + VLM client
├── scripts/
│   ├── build.sh                colcon build + rbnx codegen
│   └── start.sh                source ROS, exec atlas_bridge
└── src/                        vendored dependencies (if any)
```

## Config (passed via `Driver(CMD_INIT, config_json)`)

```json
{
  "llm_base_url":       "https://api.ofox.ai/v1",
  "llm_api_key":        "sk-...",
  "llm_model":          "google/gemini-3.1-flash-lite",
  "temperature":        0.0,
  "rotation_cam2arm":   true,
  "skip_depth":         true,
  "rgb_topic":          "",
  "camera_info_topic":  "",
  "detect_backend":     "yolo_http",
  "yolo_http_url":      "http://127.0.0.1:8770/detect",
  "yolo_http_timeout_s": 10.0,
  "yolo_conf":          0.5,
  "yolo_iou":           0.45
}
```

- `rotation_cam2arm`: rotate the image 180° before sending to the LLM when the camera is mounted opposite to the arm. (VLM backend only — the yolo backend runs on the raw frame.)
- `skip_depth`: vertical-grasp mode — bypass depth back-projection entirely. `env LLM_DETECT_SKIP_DEPTH=1` is the fallback.
- Empty topic strings mean "atlas-resolved"; defaults already match `primitive-orbbec-dabai_dcw-camera-rbnx`.
- `detect_backend`: which backend answers a call that omits `backend` (`yolo_http` or `llm`; aliases `yolo` / `vlm` accepted). Must be a backend that is actually configured.
- `yolo_http_url` / `yolo_http_timeout_s`: host GPU YOLO server endpoint and per-request timeout. An empty URL disables the YOLO route.
- `yolo_conf` / `yolo_iou`: optional detection thresholds forwarded to the YOLO server; omit them to use the server's own defaults.

## Build / run standalone

```bash
bash scripts/build.sh                           # colcon + rbnx codegen
ROBONIX_ATLAS=127.0.0.1:50051 \
    bash scripts/start.sh                       # registers, awaits Init
```

## Verification

```bash
rbnx caps | grep object_detect
# Expected: llm_detect provider with
#   robonix/service/perception/object_detect/{driver, detect_object}

# End-to-end via the atlas MCP surface:
rbnx ask "where is the paper on the desk?"

# The package intentionally has no direct ROS service surface.
```

## License

This package: MulanPSL-2.0.
