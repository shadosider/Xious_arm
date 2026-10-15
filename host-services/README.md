# 宿主侧模型服务（host-services）

**这一层为什么存在**：robonix 容器里不装深度学习依赖（没有 torch/CUDA/ultralytics，见 `containers/`）。耗算力的活（YOLO 推理、FunASR 语音识别、VLA 策略）都跑在**宿主机**上，容器内的包通过 `--net host` 用 localhost 访问它们。检测、语音、策略三条链路在宿主侧的那一半，都在这里。

## 一览

| 服务 | 端点 | 代码 | 模型 | 谁在用 |
|---|---|---|---|---|
| YOLO 检测 | `http://127.0.0.1:8770/detect`（HTTP） | `yolo/yolo_detect_server.py` | `yolo/weights/best.pt`（5.6 MB，**入库**） | 抓取技能的技能内检测（`pick_skill/detect.py` 直连）；`llm_detect` 服务的 yolo 后端 |
| FunASR 流式识别 | `ws://127.0.0.1:8771`（WebSocket） | `funasr/funasr_asr_server.py` | `funasr/model/speech_paraformer-…-online/`（849 MB，**不入库**，随仓库放在磁盘上） | `service-speech`（`SPEECH_BACKEND=custom` → `remote_asr` 转发） |
| openpi VLA 策略 | `http://127.0.0.1:8002`（HTTP） | `openpi/run_openpi_server.sh` | openpi 框架 + `pi05_piper` checkpoint（**都没随仓库带**） | vla 技能（当前不可用，见下） |
| InternVLA 导航 | `http://127.0.0.1:5801`（HTTP） | **在 skill 包内**：`packages/skills/skill-internvla-navigation-rbnx/compose.yaml`（docker 镜像） | 本机 `/home/dog/s10-models/InternVLA-N1-DualVLN`（16 G，**不入库**） | `skill-internvla-navigation-rbnx` |

外部 API（不是本地服务，只是端点加密钥）：pilot 的 LLM、`llm_detect` 的看图模型、抓取技能的 VLM 兜底、腾讯云 TTS —— 都在部署清单的 `config:` 里，真实密钥只放在 `.local.yaml`。

## 环境：一个服务一个 venv

```bash
bash host-services/yolo/setup_venv.sh      # 建 host-services/yolo/.venv
bash host-services/funasr/setup_venv.sh    # 建 host-services/funasr/.venv
```

两个服务的 torch 需求不同（YOLO 要 torch+torchvision，FunASR 要 torch+torchaudio），各建各的，省得互相牵制。setup 脚本按平台挑 torch：`/proc/device-tree/model` 里有 Orin 就用 `torch-orin.txt`（从 NVIDIA 的 Jetson 索引装——镜像站上 aarch64 的轮子没有 sm_87 的 kernel），是 Thor 就用 `torch-thor.txt`（镜像站的 cu130 轮子带 sm_110）。x86_64（开发机 / 无 GPU 的 VM）上没有设备树，落到 `uname -m` → `torch-x86.txt`（PyTorch 官方 CPU 索引的轮子）；RK3588 这类没 CUDA 的 ARM 板子用 `torch-cpu.txt`。torch 单独装一趟、索引只给一个：混进镜像站的话 uv 会挑到没 sm_87 的那份。装完脚本打印这份 torch 编了哪些 arch，跟本机 GPU 对不上会当场提示（CPU 版会直接说推理走 CPU）。

venv 可以随时删掉重建（`.venv/` 已在 .gitignore 里）。不符号链接 roboarm 的 venv（早期方案已废弃）。

## 起服务

```bash
# YOLO（:8770）
bash host-services/yolo/run_yolo_server.sh

# FunASR（:8771）；模型不在时先拉：bash host-services/funasr/fetch_funasr_model.sh
bash host-services/funasr/run_funasr_server.sh

# openpi VLA（可选；当前不可用，见下）
bash host-services/openpi/run_openpi_server.sh
```

两个 run 脚本发现 venv 不在都会自动建一次。

## 实机上踩过的（Jetson）

- **`import torch` 报 `libcudss.so.0: cannot open shared object file`**：Jetson 版 torch 要 cuDSS，而它装在 `/usr/lib/aarch64-linux-gnu/libcudss/<版本>/` —— 一个不在库搜索路径里的版本子目录。登记一次就好：
  ```bash
  ls -d /usr/lib/aarch64-linux-gnu/libcudss/*/ | sudo tee /etc/ld.so.conf.d/cudss.conf >/dev/null
  sudo ldconfig
  ```
  两个 setup 脚本装完会自己检查这一步、没做就提示你。
- **启动 robonix 之前，先让标定台放开相机**（相机和臂同一时刻只给一个步骤用）。相机被占着时，容器里的相机原语 30 秒拿不到帧 → 原语初始化失败 → 整个 boot 中止。到标定台点「结束本步骤」，或在别的页面点「强制接管」。

## 模型从哪来

| 模型 | 大小 | 状态 |
|---|---|---|
| YOLO `best.pt` | 5.6 MB | **入库** ✓ |
| FunASR paraformer-large | 849 MB | 放在 `funasr/model/`（不进 git）；丢失后用 `funasr/fetch_funasr_model.sh` 从 modelscope 重新下载 |
| InternVLA-N1-DualVLN | 16 G | 本机放在 `/home/dog/s10-models/`（不在仓库里）；换机器要另备一份，并在 `skill-internvla-navigation-rbnx/.env` 里把 `MODEL_PATH` 指过去 |
| openpi 框架 + pi05_piper checkpoint | 数百 MB + GB 级 | **没有随机器迁移**：需要重新执行 `openpi/setup_openpi.sh` 拉取框架，checkpoint 另行准备（或从旧机器拷贝），所以 VLA 这条路目前不可用 |

## 自测（不需要麦克风与相机）

```bash
# ASR：把仓库里那份模型自带的示例音频流给 :8771，看是否逐句出文本
host-services/funasr/.venv/bin/python host-services/funasr/test_asr_ws_client.py
# YOLO：POST 一张图到 :8770/detect，返回 detections 列表
curl -s -X POST --data-binary @some.jpg "http://127.0.0.1:8770/detect?conf=0.5&iou=0.45"
```

## 目录

```text
host-services/
├── yolo/{setup_venv.sh, run_yolo_server.sh, requirements.txt,
│         torch-{orin,thor,cpu,x86}.txt, yolo_detect_server.py, weights/best.pt}
├── funasr/{setup_venv.sh, run_funasr_server.sh, fetch_funasr_model.sh,
│           requirements.txt, torch-{orin,thor}.txt, funasr_asr_server.py,
│           test_asr_ws_client.py, model/…}
└── openpi/{run_openpi_server.sh, setup_openpi.sh}
```
