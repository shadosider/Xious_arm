#!/usr/bin/env bash
# SPDX-License-Identifier: MulanPSL-2.0
# 建 YOLO 检测服务自己的 venv：host-services/yolo/.venv
#
# 装两趟：
#   1) torch-{orin,thor,cpu,x86}.txt —— 按平台选。Orin 是 sm_87，PyPI/镜像站上
#      aarch64 的轮子没编它的 kernel，得从 NVIDIA 的 Jetson 索引拿；Thor 是 sm_110，
#      镜像站的 cu130 轮子就行；RK3588 和 x86_64（开发机/无 GPU 的 VM）没 CUDA，
#      走 CPU 轮子。这一份单独装，因为它的索引只给一个（各自文件里有详细注释）。
#   2) requirements.txt —— 其余依赖，走镜像站。
# 平台看 /proc/device-tree/model（只有 ARM 板子有这个文件）；x86_64 上没有设备树，
# 落到 uname -m 判成 CPU 版。已有 venv 就复用，不删。
set -euo pipefail

HS="$(cd "$(dirname "$0")" && pwd)"        # host-services/yolo
VENV="${YOLO_VENV:-$HS/.venv}"
# 缓存跟仓库放一起：Jetson 的根分区常很小（eMMC），而 uv 默认缓存写在 ~/.cache/uv，
# 下一个 3 GB 的 torch 就能把 / 撑满。默认落到 host-services/.uv-cache（两个服务共用）。
export UV_CACHE_DIR="${UV_CACHE_DIR:-$HS/../.uv-cache}"

command -v uv >/dev/null 2>&1 || { echo "error: 找不到 uv（装一个：python -m pip install uv）" >&2; exit 1; }

# 别写成 "$(tr < /proc/device-tree/model 2>/dev/null)"：重定向失败时是 shell 自己报错，
# 那时 2>/dev/null 还没生效，x86 上会白打一行 "没有那个文件或目录"。
if [ -r /proc/device-tree/model ]; then
    MODEL="$(tr -d '\0' < /proc/device-tree/model)"
else
    MODEL=""
fi
ARCH="$(uname -m)"
case "$MODEL" in
    *Orin*) TORCH="$HS/torch-orin.txt" ;;
    *Thor*) TORCH="$HS/torch-thor.txt" ;;
    *RK3588*) TORCH="$HS/torch-cpu.txt" ;;
    "")     # 没有设备树 → 不是 ARM 板子，按 CPU 架构判（x86 开发机 / VM）
        case "$ARCH" in
            x86_64|amd64) TORCH="$HS/torch-x86.txt" ;;
            *) echo "error: 认不出平台（无 /proc/device-tree/model，uname -m = '$ARCH'）" >&2
               echo "       照 torch-*.txt 里的注释手工装" >&2
               exit 1 ;;
        esac ;;
    *) echo "error: 认不出板子（/proc/device-tree/model = '$MODEL'）" >&2
       echo "       照 torch-orin.txt / torch-thor.txt 里的注释手工装" >&2
       exit 1 ;;
esac
echo "[setup] 平台：${MODEL:-$ARCH} → $(basename "$TORCH")"

# 环境里残留的索引设置会盖掉 requirements 文件里的 --index-url，清掉
unset UV_INDEX_URL UV_EXTRA_INDEX_URL || true
# 本机没有 py3.10 时 uv 会去下独立的 CPython（走 GitHub），默认挂 gh-proxy
: "${UV_PYTHON_INSTALL_MIRROR:=https://hk.gh-proxy.org/https://github.com/astral-sh/python-build-standalone/releases/download}"
export UV_PYTHON_INSTALL_MIRROR

if [ -x "$VENV/bin/python" ]; then
    echo "[setup] 复用已有环境 → $VENV"
else
    echo "[setup] 建 venv → $VENV（python 3.10）"
    uv venv --python 3.10 "$VENV"
fi

echo "[setup] 装 torch（$(basename "$TORCH")；CPU 版几百 MB，Jetson 版约 3 GB）"
uv pip install --python "$VENV/bin/python" -r "$TORCH"
echo "[setup] 装其余依赖"
uv pip install --python "$VENV/bin/python" -r "$HS/requirements.txt"

# Jetson 版 torch 要 libcudss.so.0：它装在 /usr/lib/aarch64-linux-gnu/libcudss/<版本>/ 里，
# 没登记进动态库缓存的话 import torch 直接 ImportError（这个坑踩过）
if [ -d /usr/lib/aarch64-linux-gnu/libcudss ] && ! ldconfig -p | grep -q libcudss; then
    echo "[setup] ⚠️ 系统里有 libcudss 但动态库缓存里没有，torch 会 import 失败。执行这条修一次："
    echo "[setup]    ls -d /usr/lib/aarch64-linux-gnu/libcudss/*/ | sudo tee /etc/ld.so.conf.d/cudss.conf >/dev/null && sudo ldconfig"
fi

echo "[setup] 自测"
"$VENV/bin/python" - <<'PY'
import torch, ultralytics
arch = torch.cuda.get_arch_list()
print("[setup]   torch", torch.__version__, "| cuda", torch.version.cuda, "| 编译的 arch:", arch)
print("[setup]   ultralytics", ultralytics.__version__)
if torch.cuda.is_available():
    cap = "sm_%d%d" % torch.cuda.get_device_capability(0)
    print("[setup]   GPU:", torch.cuda.get_device_name(0), cap)
    if cap not in arch:
        print("[setup]   ⚠️ 这份 torch 没有", cap, "的 kernel，跑推理会报 no kernel image is available")
else:
    print("[setup]   无可用 CUDA → 推理走 CPU（CPU 版轮子 / x86 开发机上属预期）")
PY

echo "[setup] 完成。启动：bash $HS/run_yolo_server.sh"
