#!/bin/bash
# ===================================================================
# AutoDL 环境一键部署脚本
#
# 在 AutoDL GPU 机器上运行此脚本，完成：
#   1. 安装依赖（vLLM、transformers 等）
#   2. 下载 LawLLM-7B 模型权重（通过 HF 镜像加速）
#   3. 启动 vLLM OpenAI 兼容 API 服务
#
# 使用方法:
#   chmod +x scripts/autodl/setup_autodl.sh
#   ./scripts/autodl/setup_autodl.sh           # 完整部署
#   ./scripts/autodl/setup_autodl.sh --install # 只安装依赖
#   ./scripts/autodl/setup_autodl.sh --download # 只下载模型
#   ./scripts/autodl/setup_autodl.sh --serve   # 只启动服务
# ===================================================================

set -e

# ── 配置 ──────────────────────────────────────────────────────────
MODEL_NAME="ShengbinYue/LawLLM-7B"
# HF 镜像加速（国内必用）
export HF_ENDPOINT="https://hf-mirror.com"
# vLLM 服务端口（AutoDL 会自动映射为公网可访问的代理地址）
SERVE_PORT=8000
# 模型下载到本地的路径
MODEL_DIR="/root/autodl-tmp/models/LawLLM-7B"

# ── 颜色输出 ────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[AutoDL Setup]${NC} $1"; }
warn() { echo -e "${YELLOW}[Warning]${NC} $1"; }

# ── 安装依赖 ────────────────────────────────────────────────────────
install_deps() {
    log "安装依赖..."

    # AutoDL 通常已预装 PyTorch，检查版本
    python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || {
        warn "PyTorch 未安装，正在安装..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    }

    # 安装 vLLM 和相关依赖
    pip install vllm>=0.6.0 transformers>=4.44.0
    pip install openai  # 本地测试用
    pip install FlagEmbedding sentence-transformers  # embedding/rerank 用

    log "依赖安装完成"
    nvidia-smi
}

# ── 下载模型 ────────────────────────────────────────────────────────
download_model() {
    log "下载模型: $MODEL_NAME"
    log "使用 HF 镜像: $HF_ENDPOINT"

    mkdir -p "$(dirname "$MODEL_DIR")"

    # 用 huggingface-cli 下载（支持镜像加速）
    pip install -U huggingface_hub
    huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR"

    log "模型下载完成: $MODEL_DIR"
    du -sh "$MODEL_DIR"
}

# ── 启动 vLLM 服务 ──────────────────────────────────────────────────
serve() {
    log "启动 vLLM API 服务..."

    # 检查模型是否存在
    if [ ! -d "$MODEL_DIR" ]; then
        warn "模型目录不存在: $MODEL_DIR"
        warn "请先运行: ./setup_autodl.sh --download"
        exit 1
    fi

    # 启动 vLLM OpenAI 兼容 API 服务
    # --served-model-name: 与 config.yaml 中 api.model_name 对应
    # --quantization: 如果下载的是量化版本则加此参数
    # --max-model-len: 最大序列长度，7B 模型设 4096 够用
    # --gpu-memory-utilization: 显存利用率，4090 24GB 建议 0.9
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_DIR" \
        --served-model-name "$MODEL_NAME" \
        --trust-remote-code \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.9 \
        --port "$SERVE_PORT" \
        --host 0.0.0.0

    # 服务启动后，AutoDL 会分配一个公网代理地址，格式如：
    # https://www.autodl.com/api/v1/xxxxx
    # 将该地址填入 configs/config.yaml 的 api.base_url
}

# ── 主逻辑 ──────────────────────────────────────────────────────────
case "${1:-all}" in
    --install)  install_deps ;;
    --download) download_model ;;
    --serve)    serve ;;
    all)
        install_deps
        download_model
        serve
        ;;
    *)
        echo "用法: $0 [--install|--download|--serve]"
        echo "  无参数   = 完整部署（安装+下载+启动）"
        echo "  --install  = 只安装依赖"
        echo "  --download = 只下载模型"
        echo "  --serve    = 只启动 API 服务"
        exit 1
        ;;
esac