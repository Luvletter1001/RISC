#!/usr/bin/env bash
# =============================================================================
# Per-angle TTA Evaluation Launcher (批量多模型顺序评测版)
# 
# 对每个目标角度 θ，进行 TTA 推理并评估：
#   - TTA views: θ + {0, 90, 180, 270}（旋回 canonical 坐标后 NMS 融合）
#   - 在 angle_θ 的 GT 上评估
#
# 支持批量评测多个模型，按顺序依次处理
#
# Usage:
#   bash M_Tools/analysis/run_per_angle_tta.sh [OPTIONS]
#
# Options:
#   --models MODEL1 MODEL2 ...   指定要评估的模型 (default: rtmdet_l h2rbox_v2 retinanet_msrr redet orcnn)
#   --angles ANG1 ANG2 ...      指定目标角度 (default: 000 030 060 090 120 150 180 210 240 270 300 330)
#   --gpus GPU_IDS              GPU IDs (default: 4,5,6,7)
#   --out-root DIR               输出目录 (default: 自动生成)
#   --dry-run                    仅打印配置，不执行
#   --help                       显示帮助
# =============================================================================

set -uo pipefail

# 默认参数
ROOT_DIR="${ROOT_DIR:-/data1/zcy/OpenRSD}"
PYTHON_BIN="${PYTHON_BIN:-/data/zcy/anaconda3/envs/openrsd/bin/python}"
GPUS="${GPUS:-4,5,6,7}"  # 4卡分布式
OUT_ROOT="${OUT_ROOT:-}"
DRY_RUN="${DRY_RUN:-0}"

# 解析参数
MODELS=()
ANGLES=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --angles)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ANGLES+=("$1")
                shift
            done
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --out-root)
            OUT_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="1"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL1 MODEL2 ...  指定要评估的模型 (default: rtmdet_l h2rbox_v2 retinanet_msrr redet orcnn)"
            echo "  --angles ANG1 ANG2 ...    指定目标角度 (default: 000 030 060 090 120 150 180 210 240 270 300 330)"
            echo "  --gpus GPU_IDS              GPU IDs (default: 4,5,6,7)"
            echo "  --out-root DIR              输出目录"
            echo "  --dry-run                   仅打印配置，不执行"
            echo "  --help                      显示帮助"
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown option: $1"
            exit 1
            ;;
    esac
done

# 默认模型和角度
if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=("rtmdet_l" "h2rbox_v2" "retinanet_msrr" "redet" "orcnn")
fi
if [[ ${#ANGLES[@]} -eq 0 ]]; then
    ANGLES=("000" "030" "060" "090" "120" "150" "180" "210" "240" "270" "300" "330")
fi

# 构建命令
SCRIPT="$ROOT_DIR/M_Tools/analysis/per_angle_tta_eval.py"
CMD="$PYTHON_BIN $SCRIPT --gpus $GPUS"

for model in "${MODELS[@]}"; do
    CMD="$CMD --models $model"
done

for angle in "${ANGLES[@]}"; do
    CMD="$CMD --angles $angle"
done

if [[ -n "$OUT_ROOT" ]]; then
    CMD="$CMD --out-root $OUT_ROOT"
fi

# 显示配置
echo "============================================================"
echo "Per-angle TTA Evaluation (批量多模型顺序评测)"
echo "============================================================"
echo "Root:     $ROOT_DIR"
echo "Python:   $PYTHON_BIN"
echo "Models:   ${MODELS[*]}"
echo "Angles:   ${ANGLES[*]} (${#ANGLES[@]} 个)"
echo "GPUs:     $GPUS"
echo "Output:   ${OUT_ROOT:-auto}"
echo "============================================================"

if [[ "$DRY_RUN" == "1" ]]; then
    echo ""
    echo "[DRY RUN] Command:"
    echo "$CMD"
    echo ""
    echo "可用模型列表:"
    echo "  - rtmdet_l:        Rotated RTMDet-L 3x DOTA MS"
    echo "  - h2rbox_v2:       H2RBox-V2 R50 MS-RR"
    echo "  - retinanet_msrr:  Rotated RetinaNet R50 MS-RR"
    echo "  - redet:           ReDet R50 ReFPN"
    echo "  - orcnn:           Oriented RCNN R50 FPN"
    exit 0
fi

echo ""
echo "Running..."
echo ""
eval $CMD
