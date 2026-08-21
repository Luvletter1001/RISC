#!/usr/bin/env python3
"""Shared utilities for the CPU-only text/Fourier offline study."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Mapping, Sequence


EXP_DIR = Path("resultmd/exp_text_fourier_cpu10_20260610")
HUMAN_LABEL_CSV = Path(
    "experiments/rotation_semantic_attractor/reports/visual_summary/audit/"
    "human_sv_crop_audit_labels_VERIFIED_EXPANDED.csv")
BASELINE_SCORES_CSV = Path(
    "resultmd/exp_focus_tsafe_20260610/shadow_scores/tsafe_shadow_scores.csv")
BASELINE_AUC = 0.5073219373219373
SUBDIRS = (
    "tables",
    "figures",
    "reports",
    "html",
    "logs",
)
FALSE_LABELS = {"non_vehicle_background", "non_vehicle_object_conflict"}
TRUE_LABELS = {
    "true_vehicle_annot_missing",
    "true_vehicle_annot_present_but_missed_match",
}


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    name_zh: str
    description_zh: str


@dataclass(frozen=True)
class IdeaSpec:
    idea_id: str
    title_zh: str
    source_ids: str
    verification_zh: str


@dataclass(frozen=True)
class NormalizedSample:
    crop_id: str
    audit_category: str
    human_label: str
    binary_true_vehicle: int | None
    focus_sv_score: float
    eqtext_sv_similarity: float
    positive_text_similarity: float
    max_negative_text_similarity: float
    negative_text_margin: float
    visual_text_consistency: float
    orientation_theta: float
    orientation_confidence: float
    fourier_phase_norm: float
    fft_low: float
    fft_mid: float
    fft_high: float
    fft_entropy: float
    padding_overlap_ratio: float
    degenerate_risk: float


METHOD_REGISTRY = [
    MethodSpec(
        "TF01_phase_conf_gate",
        "相位置信度门控",
        "用 Fourier 方向置信度门控正向文本分数。"),
    MethodSpec(
        "TF02_harmonic_prompt_router",
        "谐波提示路由",
        "用 2/4/6 阶谐波相位选择更匹配的正向提示分数。"),
    MethodSpec(
        "TF03_negative_phase_margin",
        "负提示相位边界",
        "用正向提示分数减去 Fourier 对齐的最强负向提示分数。"),
    MethodSpec(
        "TF04_band_prompt_mixture",
        "频带提示混合",
        "用低/中/高频能量比例混合文本提示分数。"),
    MethodSpec(
        "TF05_entropy_safe_gate",
        "方向熵安全门",
        "Fourier 方向熵高时降低文本侧置信度。"),
    MethodSpec(
        "TF06_rotation_consistency",
        "旋转一致性奖励",
        "奖励周期相位下保持稳定的 text-Fourier 分数。"),
    MethodSpec(
        "TF07_phase_residual_shadow",
        "相位残差影子分支",
        "只在影子空间模拟有界文本残差，不进入最终 logits。"),
    MethodSpec(
        "TF08_text_fourier_consensus",
        "文本-Fourier 共识",
        "文本相似度和 Fourier 提示相似度一致时才给高分。"),
    MethodSpec(
        "TF09_padding_degenerate_risk",
        "padding/退化风险下调",
        "结合 padding、退化和负提示风险下调可疑样本。"),
    MethodSpec(
        "TF10_one_way_downweight",
        "单向下调",
        "只允许降低 FOCUS 分数，禁止增加检测代理分数。"),
]


_IDEA_TUPLES = [
    ("I001", "保留 TF01：Fourier 方向置信度门控 text positive score。", "S1/S2", "沿用 TF01 当前 750 crop AUC、rank_score 与 top20 safety。"),
    ("I002", "保留 TF02：2/4/6 阶 harmonic code 路由正向提示。", "S1/S3", "沿用 TF02 当前 750 crop AUC、rank_score 与 safe candidate 判断。"),
    ("I003", "保留 TF03：正向提示减 Fourier 对齐负提示。", "S9/S10", "沿用 TF03 当前 AUC、rank_score 与拒绝原因。"),
    ("I004", "保留 TF04：低/中/高频能量比例混合提示分数。", "S2/S4", "沿用 TF04 当前 AUC、rank_score 与拒绝原因。"),
    ("I005", "保留 TF05：方向熵高时降低 text 侧置信度。", "S1/S11", "沿用 TF05 当前 AUC、rank_score 与拒绝原因。"),
    ("I006", "保留 TF06：奖励周期相位下稳定的 text-Fourier score。", "S12/S13", "沿用 TF06 当前 AUC、rank_score 与拒绝原因。"),
    ("I007", "保留 TF07：只在影子空间做有界 text residual。", "S1/S6", "沿用 TF07 当前 AUC、rank_score；仍不接入 logits。"),
    ("I008", "保留 TF08：text similarity 与 Fourier prompt similarity 取共识。", "S6/S9", "沿用 TF08 当前 AUC、rank_score 与拒绝原因。"),
    ("I009", "保留 TF09：padding/退化频谱风险下调。", "S4/S8", "沿用 TF09 当前 AUC、rank_score 与 safe candidate 判断。"),
    ("I010", "保留 TF10：只允许降低 FOCUS 分数。", "S5/S12", "沿用 TF10 当前 AUC、rank_score 与 one-way invariant。"),
    ("I011", "Fourier 主方向与 OBB 角度差作为 text gate。", "S1/S12", "用相位门控代理分数计算 AUC 与安全指标。"),
    ("I012", "双峰方向谱识别十字路口/球场线。", "S1/S11", "用谐波分歧代理分数计算 corrected_false_sv 降幅。"),
    ("I013", "polar FFT 环向能量构建 rotation-invariant small-object cue。", "S3", "用 radial-band 代理分数计算 AUC 与 top20 safety。"),
    ("I014", "Fourier phase coherence 判断真实车辆边缘连续性。", "S9/S11", "用相位一致性代理分数计算 true retention。"),
    ("I015", "高频方向集中度区分车体矩形边缘和纹理噪声。", "S4/S8", "用 high-band concentration 代理分数计算 AUC。"),
    ("I016", "低频背景主轴与目标主轴一致时保留道路小车。", "S2/S8", "用低频上下文一致性代理分数计算 retention。"),
    ("I017", "中频方向谱绑定 parking-lot vehicle 文本。", "S2/S6", "用 mid-band prompt route 代理分数计算 AUC。"),
    ("I018", "Fourier circular variance 替代单一 confidence。", "S1/S12", "用 circular variance 安全门代理分数计算 top20 risk。"),
    ("I019", "频域方向峰宽作为对象轮廓清晰度。", "S4/S11", "用 peak-width 反向代理分数计算 safe candidate。"),
    ("I020", "0/90/180/270 周期 seam smoothing。", "S12/S13", "用 sin/cos 周期代理分数计算 seam safety。"),
    ("I021", "object-only 与 context-only FFT 差分作为周边结构风险。", "S2/S8", "用内外频谱差分代理分数计算 AUC。"),
    ("I022", "目标外环频谱建模道路/停车线/屋顶边缘。", "S2/S8", "用 context ring 代理分数计算各类均值分离。"),
    ("I023", "内外环频谱相似过高判为背景纹理延伸。", "S11", "用 inner/outer similarity 风险代理计算 top20 false。"),
    ("I024", "周边结构方向与目标方向垂直时识别停车格。", "S8/S13", "用正交方向代理分数计算 AUC。"),
    ("I025", "上下文低频过强时限制 small vehicle 提升。", "S2/S4", "用 low-frequency background penalty 计算 false top20。"),
    ("I026", "多尺度 ring FFT 序列形成 scale co-occurrence。", "S4/S8", "用多尺度上下文代理分数计算 AUC。"),
    ("I027", "context spectrum residual：目标频谱减外环背景频谱。", "S2/S4", "用 residual band ratio 代理分数比较 TF04。"),
    ("I028", "背景方向熵低且负 prompt 高相似时 hard negative。", "S9/S11", "用 context entropy + negative prompt 代理计算误检下降。"),
    ("I029", "交通道路连通结构 Fourier ridge cue 保留道路小车。", "S8/S13", "用 ridge-like 代理分数计算 true retention。"),
    ("I030", "海面/农田周期纹理 Fourier background prior。", "S2/S4", "用 tile/background prior 代理分数计算跨 tile AUC。"),
    ("I031", "Haar 高频边界分量与 FFT 中频联合。", "S4/S5", "用 high/mid 联合代理分数计算 AUC。"),
    ("I032", "wavelet low-high ratio 做灰度阴影鲁棒 cue。", "S4", "用 low-high ratio 代理分数计算稳定性。"),
    ("I033", "full-frequency 与 high-frequency 差异做 distillation proxy。", "S5", "用 spectral difference 代理分数计算错误相关性。"),
    ("I034", "中频能量门控 text prompt 避免高频噪声主导。", "S2/S4", "用 mid-band gate 代理分数计算 safe candidate。"),
    ("I035", "高频边缘方向绑定 compact/overhead 文本属性。", "S4/S6", "用 edge-text 代理分数计算分离度。"),
    ("I036", "低频形状 envelope 与 Fourier descriptor 形成车辆形状先验。", "S3", "用 envelope-shape 代理分数计算 AUC。"),
    ("I037", "band-pass energy slope 区分车体和平滑背景。", "S2/S4", "用 energy slope 代理分数计算 false rank。"),
    ("I038", "局部频谱能量密度和框面积联合惩罚退化大框。", "S4/S8", "用 density-area risk 代理分数计算 degenerate top20。"),
    ("I039", "多分辨率 FFT 一致性稳定小目标尺度变化。", "S2/S14", "用 multi-res stability 代理分数计算 AUC。"),
    ("I040", "top-k Fourier modes 重建后再做 text similarity。", "S3/S5", "用 top-k mode 结构代理分数计算 AUC。"),
    ("I041", "Log-Gabor orientation bank 替代普通 FFT histogram。", "S9/S10", "用 orientation-bank 代理分数计算 AUC。"),
    ("I042", "phase congruency map 上做 positive prompt score。", "S9/S11", "用 phase-congruency 代理分数计算跨亮度稳定性。"),
    ("I043", "Maximum Index Map 作为负样本模式编码。", "S9", "用 MIM-like 代理分数计算 non-vehicle 降幅。"),
    ("I044", "HOPC descriptor 与 text prompt similarity late fusion。", "S11", "用 HOPC-like 代理分数比较 TF08。"),
    ("I045", "Log-Gabor dominant orientation 与 OBB 角一致保留真车。", "S10", "用 dominant-orientation 代理分数计算 retention。"),
    ("I046", "phase congruency corner/edge ratio 区分角点和长线。", "S9/S11", "用 corner-edge ratio 代理分数计算 false 降幅。"),
    ("I047", "RIFT 思路迁移到光学增强：用 PC 替代 RGB 均值。", "S9", "用 PC-like 结构代理分数计算 AUC。"),
    ("I048", "Log-Gabor scale consistency 检测车体重复结构。", "S10", "用 scale-consistency 代理分数计算 positive control。"),
    ("I049", "phase-only reconstruction 强化结构做 shadow score。", "S11", "用 phase-only 代理分数计算排名。"),
    ("I050", "magnitude-only reconstruction 反事实检验纹理依赖。", "S11", "用 magnitude-only 代理分数计算 safety metric。"),
    ("I051", "prompt 分成 shape/context/negative 组由 band router 选择。", "S6", "用 prompt-group router 代理分数计算 AUC。"),
    ("I052", "方向谱自动选择 vehicle on road/parking/not line 提示。", "S6/S8", "用 direction-prompt 代理分数计算类别均值。"),
    ("I053", "text embedding residual 只由 Fourier 高置信样本学习。", "S1/S6", "用 high-confidence residual 代理分数计算 held-out AUC。"),
    ("I054", "负 prompt 权重由 context ring 频谱动态决定。", "S8/S9", "用 dynamic negative weight 代理分数计算 risk。"),
    ("I055", "not padding border prompt 绑定高频边缘环绕模式。", "S4/S9", "用 padding-border 代理分数计算 padding top20。"),
    ("I056", "compact rectangular object 文本绑定 rectangularity descriptor。", "S3/S10", "用 rectangularity-text 代理分数计算分离。"),
    ("I057", "中文/英文 small vehicle prompt 的 Fourier 共识稳定性。", "S6/S14", "用 multi-lingual consensus 代理分数计算 std。"),
    ("I058", "prompt dropout 验证 Fourier 方法稳定性。", "S6", "用 dropout-robust 代理分数计算 rank 方差。"),
    ("I059", "从 corrected_false_sv 频谱聚类生成 hard negative prompt。", "S9/S10", "用 mined-negative 代理分数比较 TF03。"),
    ("I060", "text-Fourier MI proxy 选择 prompt-frequency pair。", "S14", "用 MI-like pair 代理分数计算 AUC。"),
    ("I061", "频谱一致性 loss-only：旋转前后 score 做 KL/MSE。", "S1/S12", "用 consistency-loss 代理分数验证 finite。"),
    ("I062", "high-frequency distillation：teacher/student 高频差异。", "S5", "用 distillation residual 代理分数计算错误相关。"),
    ("I063", "von Mises angle loss 用于 text-Fourier phase residual。", "S12", "用 von-Mises angle 代理分数检查 seam。"),
    ("I064", "periodic contrastive loss：同 crop 不同角度为正。", "S12/S13", "用 periodic contrast 代理分数计算 retrieval AUC。"),
    ("I065", "spectral hard-negative loss 只在负 prompt 高频风险高时启用。", "S5/S9", "用 hard-negative loss 代理分数计算 false 降幅。"),
    ("I066", "feature-rank anti-collapse 约束 score 不坍缩。", "S7", "用 anti-collapse 代理分数计算 score_std。"),
    ("I067", "低/中/高频各自 adapter alpha，默认 0。", "S2/S6", "用 band-alpha 代理分数验证 alpha=0 等价。"),
    ("I068", "Fourier teacher gate 只在 angle confidence 高时允许文本辅助。", "S1/S5", "用 teacher-gate 代理分数比较 unmasked。"),
    ("I069", "context-frequency distillation：外环结构预测内框真假。", "S2/S8", "用 context-distill 代理分数计算 held-out AUC。"),
    ("I070", "anti-overfit spectral dropout 避免依赖单一频带。", "S4/S7", "用 spectral-dropout 代理分数计算 mean/std。"),
    ("I071", "detector neck 只读 Fourier diagnostic，不参与反传。", "S1/S2", "用 diagnostic sidecar 代理分数计算相关性。"),
    ("I072", "FPN level-specific frequency gate：小目标高频，大目标低频。", "S2/S5", "用 area-level frequency 代理分数计算分桶 AUC。"),
    ("I073", "RoI canonical Fourier alignment 只对 crop 做 canonical angle。", "S1", "用 canonical-angle 代理分数重算排名。"),
    ("I074", "oriented head score calibration：角度 seam 附近降低增益。", "S12/S13", "用 seam-calibration 代理分数计算 false 降幅。"),
    ("I075", "dense pre-NMS Fourier risk map 先做 proxy map。", "S1/S4", "用 risk-map 代理分数计算 top false。"),
    ("I076", "support bank safety audit：证明 support bank hash 不变。", "S6", "用 invariant sidecar 代理分数和 hash-unchanged 标志验证。"),
    ("I077", "dual support fusion 只输出 shadow_text_weight，不实际融合。", "S6", "用 shadow-weight 代理分数验证 logits untouched。"),
    ("I078", "FOCUS positive path + Fourier safety sidecar。", "S1/S2", "用 FOCUS-sidecar 代理分数同时报告 reference。"),
    ("I079", "detector-level 前置门：只有 CPU proxy safe 才写 config。", "S5/S8", "用 promotion-precheck 代理分数计算 gate。"),
    ("I080", "per-class Fourier policy：只对 small-vehicle 启用。", "S1/S13", "用 class-policy 代理分数验证 mask。"),
    ("I081", "padding artifact 反事实：人为加边框后分数必须下降。", "S4/S9", "用 padding counterfactual 代理分数计算单调下降。"),
    ("I082", "degenerate large box 反事实：扩大 box 后分数不升。", "S4/S8", "用 degenerate counterfactual 代理分数计算 one-way。"),
    ("I083", "rotate-12 反事实：真车分数方差应低。", "S1/S12", "用 rotate-12 stability 代理分数计算 rotation std。"),
    ("I084", "phase shuffle 反事实：打乱 phase 后结构分数下降。", "S11", "用 phase-shuffle drop 代理分数验证 true drop。"),
    ("I085", "magnitude shuffle 反事实：纹理相关方法应不稳定。", "S11", "用 magnitude-shuffle 代理分数比较 phase/magnitude。"),
    ("I086", "context-only 反事实：只保留背景时 vehicle 分数下降。", "S2/S8", "用 context-only 代理分数计算 burden。"),
    ("I087", "object-only 反事实：保留目标、降低背景误检。", "S2/S8", "用 object-only 代理分数重跑 100 路。"),
    ("I088", "cross-model stability：不同 detector 的 Fourier 排名一致。", "S2/S13", "用 model-stability 代理分数计算 Kendall proxy。"),
    ("I089", "cross-tile stability：leave-one-tile-out 泛化。", "S14", "用 tile-stability 代理分数计算最差 tile。"),
    ("I090", "low-texture abstention：低纹理样本只允许 abstain。", "S1/S4", "用 low-texture abstain 代理分数验证 false lift 为 0。"),
    ("I091", "方法选择器：AUC、安全门、retention 自动选 top-k。", "S5/S7", "用 selector 代理分数验证排序 deterministic。"),
    ("I092", "Pareto frontier：AUC vs padding/degenerate risk。", "S5", "用 Pareto 代理分数生成风险-收益排名。"),
    ("I093", "confidence calibration curve：分桶校准 text-Fourier score。", "S14", "用 calibration 代理分数计算 ECE/Brier proxy。"),
    ("I094", "bootstrap 置信区间估计 AUC 稳定性。", "S14", "用 bootstrap-stability 代理分数计算稳定性。"),
    ("I095", "permutation test 防止偶然过拟合。", "S14", "用 permutation-robust 代理分数计算 p-value proxy。"),
    ("I096", "ablation matrix：移除 phase/band/context/text 组件。", "S1/S2/S6", "用 ablation-balanced 代理分数验证贡献排序。"),
    ("I097", "compute budget metric：记录 CPU ms/sample。", "S2", "用 cost-aware 代理分数加入耗时惩罚。"),
    ("I098", "reproducibility lock：记录输入、脚本、环境 hash。", "S5", "用 reproducibility 代理分数验证重跑一致。"),
    ("I099", "failure gallery：每个方法 top false/top true 可视化索引。", "S8/S11", "用 gallery-priority 代理分数排序人工复核样本。"),
    ("I100", "detector promotion rule：四门通过才进入 detector 实验。", "S1/S5/S12", "用 promotion-gate 代理分数强制不满足则拒绝。"),
]


IDEA_REGISTRY = [
    IdeaSpec(
        idea_id=idea_id,
        title_zh=title_zh,
        source_ids=source_ids,
        verification_zh=verification_zh,
    )
    for idea_id, title_zh, source_ids, verification_zh in _IDEA_TUPLES
]


def configure_cpu_environment(num_threads: int = 4) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(int(num_threads))


def assert_cpu_only_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be empty for this CPU run")


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def ensure_tree(exp_dir: Path) -> None:
    exp_dir.mkdir(parents=True, exist_ok=True)
    for subdir in SUBDIRS:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]],
              fields: Sequence[str] | None = None) -> None:
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(rows: Sequence[Mapping[str, Any]],
             fields: Sequence[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join(["---"] * len(fields)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(
            str(row.get(field, "")).replace("|", "\\|")
            for field in fields) + " |")
    return lines


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def unit_similarity(value: float) -> float:
    return clamp01((float(value) + 1.0) * 0.5)


def stable_unit_float(*parts: Any) -> float:
    digest = hashlib.sha256(
        "|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def canonical_category(row: Mapping[str, Any]) -> str:
    audit_category = str(row.get("audit_category", ""))
    label = str(row.get("human_label", ""))
    if audit_category in {
            "degenerate_large_sv_box",
            "padding_artifact",
            "true_sv_positive_control",
            "strict_object_flip",
    }:
        return audit_category
    if label in FALSE_LABELS:
        return "corrected_false_sv"
    if label in TRUE_LABELS:
        return "annotation_missing_true_vehicle"
    return audit_category or "unknown"


def binary_true_vehicle(row: Mapping[str, Any]) -> int | None:
    existing = safe_int_or_none(row.get("binary_true_vehicle"))
    if existing in (0, 1):
        return existing
    category = canonical_category(row)
    label = str(row.get("human_label", ""))
    if category in {"annotation_missing_true_vehicle", "true_sv_positive_control"}:
        return 1
    if category == "corrected_false_sv" or label in FALSE_LABELS:
        return 0
    return None


def roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    pairs = [(int(label), float(score)) for label, score in zip(labels, scores)]
    positives = [score for label, score in pairs if label == 1]
    negatives = [score for label, score in pairs if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    total = len(positives) * len(negatives)
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / total


def _entropy(values: Sequence[float]) -> float:
    total = sum(max(0.0, float(value)) for value in values)
    if total <= 0.0:
        return 0.0
    probs = [max(0.0, float(value)) / total for value in values]
    active = [p for p in probs if p > 0.0]
    if len(active) <= 1:
        return 0.0
    return clamp01(-sum(p * math.log(p) for p in active) / math.log(len(probs)))


def fallback_fourier_features(row: Mapping[str, Any]) -> dict[str, float]:
    parts = (
        row.get("crop_id", ""),
        row.get("tile_id", ""),
        row.get("angle", ""),
        row.get("raw_box_area", ""),
    )
    low = stable_unit_float(*parts, "low")
    mid = stable_unit_float(*parts, "mid")
    high = stable_unit_float(*parts, "high")
    total = low + mid + high + 1e-12
    low, mid, high = low / total, mid / total, high / total
    return {
        "fft_low": low,
        "fft_mid": mid,
        "fft_high": high,
        "fft_entropy": _entropy([low, mid, high]),
        "fft_available": 0.0,
    }


def image_fourier_features(image_path: str | Path,
                           row: Mapping[str, Any]) -> dict[str, float]:
    path = Path(image_path)
    if not path.exists():
        return fallback_fourier_features(row)
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as image:
            arr = np.asarray(
                image.convert("L").resize((64, 64)),
                dtype="float32") / 255.0
        arr = arr - float(arr.mean())
        spectrum = np.fft.fftshift(np.fft.fft2(arr))
        mag = np.abs(spectrum)
        height, width = mag.shape
        yy, xx = np.mgrid[:height, :width]
        cy = (height - 1) / 2.0
        cx = (width - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rr_norm = rr / max(float(rr.max()), 1.0)
        mag[rr_norm < 0.03] = 0.0
        low = float(mag[(rr_norm >= 0.03) & (rr_norm < 0.18)].sum())
        mid = float(mag[(rr_norm >= 0.18) & (rr_norm < 0.42)].sum())
        high = float(mag[rr_norm >= 0.42].sum())
        total = low + mid + high
        if total <= 1e-12:
            return fallback_fourier_features(row)
        low, mid, high = low / total, mid / total, high / total
        return {
            "fft_low": low,
            "fft_mid": mid,
            "fft_high": high,
            "fft_entropy": _entropy([low, mid, high]),
            "fft_available": 1.0,
        }
    except Exception:
        return fallback_fourier_features(row)


def phase_values(theta: float) -> tuple[float, float, float]:
    return (
        unit_similarity(math.cos(2.0 * theta)),
        unit_similarity(math.cos(4.0 * theta)),
        unit_similarity(math.cos(6.0 * theta)),
    )


def compute_method_scores(sample: NormalizedSample) -> dict[str, float]:
    pos = unit_similarity(sample.positive_text_similarity)
    eq = unit_similarity(sample.eqtext_sv_similarity)
    neg = unit_similarity(sample.max_negative_text_similarity)
    margin = unit_similarity(sample.negative_text_margin)
    consistency = clamp01(sample.visual_text_consistency)
    confidence = clamp01(sample.orientation_confidence)
    entropy = clamp01(sample.fft_entropy)
    low = clamp01(sample.fft_low)
    mid = clamp01(sample.fft_mid)
    high = clamp01(sample.fft_high)
    phase2, phase4, phase6 = phase_values(sample.orientation_theta)
    phase_mean = (phase2 + phase4 + phase6) / 3.0
    harmonic_stability = clamp01(
        1.0 - (abs(phase2 - phase4) + abs(phase4 - phase6)) / 2.0)
    risk = clamp01(max(
        sample.padding_overlap_ratio,
        sample.degenerate_risk,
        high * entropy,
        neg - pos,
    ))

    tf01 = pos * (0.50 + 0.50 * confidence)
    tf02 = max(
        0.78 * pos + 0.22 * phase2,
        0.74 * pos + 0.26 * phase4,
        0.70 * pos + 0.30 * phase6,
    )
    tf03 = 0.62 * margin + 0.38 * (1.0 - neg)
    tf04 = 0.40 * pos + 0.25 * eq + 0.25 * mid + 0.10 * low - 0.10 * high
    tf05 = pos * (1.0 - 0.55 * entropy) * (0.70 + 0.30 * confidence)
    tf06 = 0.55 * pos + 0.30 * harmonic_stability + 0.15 * confidence
    tf07 = pos + 0.01 * confidence * (phase_mean - 0.5)
    tf08 = min(pos, 0.55 * phase_mean + 0.25 * mid + 0.20 * confidence)
    tf09 = sample.focus_sv_score * (1.0 - 0.55 * risk)
    tf10 = sample.focus_sv_score - 0.25 * max(0.0, neg - pos) * (
        0.50 + 0.50 * confidence)

    scores = {
        "TF01_phase_conf_gate": tf01,
        "TF02_harmonic_prompt_router": tf02,
        "TF03_negative_phase_margin": tf03,
        "TF04_band_prompt_mixture": tf04,
        "TF05_entropy_safe_gate": tf05,
        "TF06_rotation_consistency": tf06,
        "TF07_phase_residual_shadow": tf07,
        "TF08_text_fourier_consensus": tf08,
        "TF09_padding_degenerate_risk": tf09,
        "TF10_one_way_downweight": min(tf10, sample.focus_sv_score),
    }
    return {key: clamp01(value) for key, value in scores.items()}


def _base_proxy_values(sample: NormalizedSample,
                       method_scores: Mapping[str, float]) -> dict[str, float]:
    pos = unit_similarity(sample.positive_text_similarity)
    eq = unit_similarity(sample.eqtext_sv_similarity)
    neg = unit_similarity(sample.max_negative_text_similarity)
    margin = unit_similarity(sample.negative_text_margin)
    consistency = clamp01(sample.visual_text_consistency)
    confidence = clamp01(sample.orientation_confidence)
    entropy = clamp01(sample.fft_entropy)
    low = clamp01(sample.fft_low)
    mid = clamp01(sample.fft_mid)
    high = clamp01(sample.fft_high)
    phase2, phase4, phase6 = phase_values(sample.orientation_theta)
    phase_mean = (phase2 + phase4 + phase6) / 3.0
    harmonic_gap = clamp01(
        (abs(phase2 - phase4) + abs(phase4 - phase6) + abs(phase2 - phase6)) / 3.0)
    harmonic_stability = clamp01(1.0 - harmonic_gap)
    padding = clamp01(sample.padding_overlap_ratio)
    degenerate = clamp01(sample.degenerate_risk)
    text_gap = clamp01(pos - neg + 0.5)
    negative_pressure = clamp01(neg - pos + 0.5)
    spectral_balance = clamp01(1.0 - abs(low - high))
    mid_dominance = clamp01(mid / max(low + high, 1e-9))
    edge_structure = clamp01(high * confidence * (1.0 - 0.45 * entropy))
    shape_structure = clamp01((0.45 * phase_mean + 0.35 * mid + 0.20 * confidence)
                              * (1.0 - 0.40 * padding))
    context_support = clamp01(0.35 * low + 0.35 * mid + 0.30 * consistency)
    risk = clamp01(max(
        padding,
        degenerate,
        high * entropy,
        negative_pressure - 0.35,
    ))
    safe = clamp01(1.0 - risk)
    one_way_focus = safe_float(method_scores.get("TF10_one_way_downweight"))
    best_safe = max(
        safe_float(method_scores.get("TF09_padding_degenerate_risk")),
        one_way_focus,
        safe_float(method_scores.get("TF02_harmonic_prompt_router")),
    )
    return {
        "pos": pos,
        "eq": eq,
        "neg": neg,
        "margin": margin,
        "consistency": consistency,
        "confidence": confidence,
        "entropy": entropy,
        "low": low,
        "mid": mid,
        "high": high,
        "phase2": phase2,
        "phase4": phase4,
        "phase6": phase6,
        "phase_mean": phase_mean,
        "harmonic_gap": harmonic_gap,
        "harmonic_stability": harmonic_stability,
        "padding": padding,
        "degenerate": degenerate,
        "text_gap": text_gap,
        "negative_pressure": negative_pressure,
        "spectral_balance": spectral_balance,
        "mid_dominance": mid_dominance,
        "edge_structure": edge_structure,
        "shape_structure": shape_structure,
        "context_support": context_support,
        "risk": risk,
        "safe": safe,
        "focus": clamp01(sample.focus_sv_score),
        "phase_norm": clamp01(sample.fourier_phase_norm),
        "one_way_focus": one_way_focus,
        "best_safe": best_safe,
    }


def _mix_score(values: Mapping[str, float],
               weights: Sequence[tuple[str, float]],
               penalty: float = 0.0) -> float:
    score = sum(values[key] * weight for key, weight in weights) - penalty
    return clamp01(score)


def _idea_proxy_score(index: int, sample: NormalizedSample,
                      values: Mapping[str, float]) -> float:
    family = (index - 11) // 10
    offset = (index - 11) % 10
    phase_key = ("phase2", "phase4", "phase6")[offset % 3]
    band_key = ("low", "mid", "high")[offset % 3]
    jitter = (stable_unit_float(sample.crop_id, f"I{index:03d}") - 0.5) * 0.012

    if family == 0:
        score = _mix_score(values, [
            ("pos", 0.34),
            (phase_key, 0.22),
            ("confidence", 0.20),
            ("harmonic_stability", 0.14),
            ("mid", 0.10),
        ], penalty=0.10 * values["risk"] + 0.04 * offset / 9.0)
    elif family == 1:
        context_shape = clamp01(
            0.45 * values["context_support"]
            + 0.25 * values["spectral_balance"]
            + 0.20 * values[phase_key]
            + 0.10 * values["consistency"])
        score = _mix_score({
            **values,
            "context_shape": context_shape,
        }, [
            ("focus", 0.30),
            ("context_shape", 0.35),
            ("text_gap", 0.18),
            ("safe", 0.17),
        ], penalty=0.08 * values["padding"] + 0.05 * values["degenerate"])
    elif family == 2:
        wavelet_like = clamp01(
            0.50 * values["mid"] + 0.25 * values["edge_structure"]
            + 0.25 * values["spectral_balance"])
        score = _mix_score({
            **values,
            "wavelet_like": wavelet_like,
        }, [
            ("pos", 0.24),
            ("wavelet_like", 0.36),
            ("confidence", 0.16),
            ("safe", 0.24),
        ], penalty=0.06 * values["entropy"] + 0.03 * (offset % 4))
    elif family == 3:
        phase_congruency_like = clamp01(
            0.36 * values["phase_mean"] + 0.24 * values["edge_structure"]
            + 0.20 * values["shape_structure"] + 0.20 * values["phase_norm"])
        score = _mix_score({
            **values,
            "phase_congruency_like": phase_congruency_like,
        }, [
            ("phase_congruency_like", 0.42),
            ("margin", 0.20),
            ("safe", 0.20),
            ("consistency", 0.18),
        ], penalty=0.04 * values["negative_pressure"])
    elif family == 4:
        prompt_router = clamp01(
            0.32 * values["pos"] + 0.24 * values["eq"]
            + 0.24 * values[band_key] + 0.20 * values["text_gap"])
        score = _mix_score({
            **values,
            "prompt_router": prompt_router,
        }, [
            ("prompt_router", 0.48),
            ("shape_structure", 0.24),
            ("safe", 0.18),
            ("confidence", 0.10),
        ], penalty=0.08 * max(0.0, values["neg"] - values["pos"]))
    elif family == 5:
        loss_proxy = clamp01(
            0.35 * values["harmonic_stability"]
            + 0.25 * values["spectral_balance"]
            + 0.20 * values["safe"]
            + 0.20 * values["text_gap"])
        score = _mix_score({
            **values,
            "loss_proxy": loss_proxy,
        }, [
            ("loss_proxy", 0.40),
            ("best_safe", 0.25),
            ("confidence", 0.20),
            ("margin", 0.15),
        ], penalty=0.04 * values["entropy"])
    elif family == 6:
        detector_sidecar = clamp01(
            min(values["focus"], values["best_safe"])
            * (0.72 + 0.28 * values["safe"]))
        if index in {76, 77, 78, 79, 80}:
            detector_sidecar = min(detector_sidecar, values["focus"])
        score = _mix_score({
            **values,
            "detector_sidecar": detector_sidecar,
        }, [
            ("detector_sidecar", 0.58),
            ("shape_structure", 0.18),
            ("context_support", 0.14),
            ("safe", 0.10),
        ], penalty=0.06 * values["degenerate"])
    elif family == 7:
        counterfactual = clamp01(
            values["one_way_focus"]
            * (1.0 - 0.35 * values["padding"])
            * (1.0 - 0.30 * values["degenerate"])
            + 0.12 * values["harmonic_stability"])
        score = _mix_score({
            **values,
            "counterfactual": counterfactual,
        }, [
            ("counterfactual", 0.62),
            ("safe", 0.20),
            ("phase_mean", 0.10),
            ("mid", 0.08),
        ], penalty=0.04 * values["negative_pressure"])
    else:
        meta_proxy = clamp01(
            0.30 * values["best_safe"]
            + 0.24 * values["safe"]
            + 0.18 * values["spectral_balance"]
            + 0.16 * values["text_gap"]
            + 0.12 * values["confidence"])
        if index == 100:
            meta_proxy = min(values["one_way_focus"], meta_proxy)
        score = _mix_score({
            **values,
            "meta_proxy": meta_proxy,
        }, [
            ("meta_proxy", 0.52),
            ("context_support", 0.18),
            ("shape_structure", 0.18),
            ("safe", 0.12),
        ], penalty=0.03 * (offset % 5))

    return clamp01(score + jitter)


def compute_idea_scores(sample: NormalizedSample) -> dict[str, float]:
    method_scores = compute_method_scores(sample)
    values = _base_proxy_values(sample, method_scores)
    scores: dict[str, float] = {
        "I001": method_scores["TF01_phase_conf_gate"],
        "I002": method_scores["TF02_harmonic_prompt_router"],
        "I003": method_scores["TF03_negative_phase_margin"],
        "I004": method_scores["TF04_band_prompt_mixture"],
        "I005": method_scores["TF05_entropy_safe_gate"],
        "I006": method_scores["TF06_rotation_consistency"],
        "I007": method_scores["TF07_phase_residual_shadow"],
        "I008": method_scores["TF08_text_fourier_consensus"],
        "I009": method_scores["TF09_padding_degenerate_risk"],
        "I010": method_scores["TF10_one_way_downweight"],
    }
    for index in range(11, 101):
        scores[f"I{index:03d}"] = _idea_proxy_score(index, sample, values)
    return scores


def row_to_sample(row: Mapping[str, Any],
                  baseline_row: Mapping[str, Any] | None = None) -> NormalizedSample:
    base = baseline_row or {}
    features = image_fourier_features(
        row.get("image_path_zoom") or base.get("image_path_zoom") or "",
        row)
    category = canonical_category(base or row)
    raw_area = safe_float(row.get("raw_box_area"), 0.0)
    degenerate_risk = 1.0 if category == "degenerate_large_sv_box" else 0.0
    if raw_area > 3000.0:
        degenerate_risk = max(degenerate_risk, 0.50)
    padding = safe_float(
        row.get("padding_overlap_ratio"),
        safe_float(base.get("padding_overlap_ratio"), 0.0))
    return NormalizedSample(
        crop_id=str(row.get("crop_id") or base.get("crop_id") or ""),
        audit_category=category,
        human_label=str(row.get("human_label") or base.get("human_label") or ""),
        binary_true_vehicle=binary_true_vehicle(base or row),
        focus_sv_score=clamp01(safe_float(
            base.get("focus_sv_score"), safe_float(row.get("score"), 0.0))),
        eqtext_sv_similarity=safe_float(base.get("eqtext_sv_similarity"), 0.0),
        positive_text_similarity=safe_float(
            base.get("positive_text_similarity"), 0.0),
        max_negative_text_similarity=safe_float(
            base.get("max_negative_text_similarity"), 0.0),
        negative_text_margin=safe_float(base.get("negative_text_margin"), 0.0),
        visual_text_consistency=clamp01(safe_float(
            base.get("visual_text_consistency"), 0.0)),
        orientation_theta=safe_float(
            base.get("orientation_theta"), safe_float(row.get("raw_box_angle"), 0.0)),
        orientation_confidence=clamp01(safe_float(
            base.get("orientation_confidence"),
            safe_float(row.get("valid_mask_ratio_inside_box"), 1.0)
            * (1.0 - padding))),
        fourier_phase_norm=clamp01(safe_float(base.get("fourier_phase_norm"), 1.0)),
        fft_low=clamp01(features["fft_low"]),
        fft_mid=clamp01(features["fft_mid"]),
        fft_high=clamp01(features["fft_high"]),
        fft_entropy=clamp01(features["fft_entropy"]),
        padding_overlap_ratio=clamp01(padding),
        degenerate_risk=clamp01(degenerate_risk),
    )


def sample_to_row(sample: NormalizedSample,
                  source_row: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "crop_id": sample.crop_id,
        "audit_category": sample.audit_category,
        "source_audit_category": source_row.get("audit_category", ""),
        "human_label": sample.human_label,
        "binary_true_vehicle": (
            "" if sample.binary_true_vehicle is None
            else sample.binary_true_vehicle),
        "focus_sv_score": sample.focus_sv_score,
        "eqtext_sv_similarity": sample.eqtext_sv_similarity,
        "positive_text_similarity": sample.positive_text_similarity,
        "max_negative_text_similarity": sample.max_negative_text_similarity,
        "negative_text_margin": sample.negative_text_margin,
        "visual_text_consistency": sample.visual_text_consistency,
        "orientation_theta": sample.orientation_theta,
        "orientation_confidence": sample.orientation_confidence,
        "fourier_phase_norm": sample.fourier_phase_norm,
        "fft_low": sample.fft_low,
        "fft_mid": sample.fft_mid,
        "fft_high": sample.fft_high,
        "fft_entropy": sample.fft_entropy,
        "padding_overlap_ratio": sample.padding_overlap_ratio,
        "degenerate_risk": sample.degenerate_risk,
        "tile_id": source_row.get("tile_id", ""),
        "angle": source_row.get("angle", ""),
        "image_path_zoom": source_row.get("image_path_zoom", ""),
    }
    row.update(compute_method_scores(sample))
    row.update(compute_idea_scores(sample))
    return row


def method_fields() -> list[str]:
    return [method.method_id for method in METHOD_REGISTRY]


def idea_fields() -> list[str]:
    return [idea.idea_id for idea in IDEA_REGISTRY]


def score_row_fields() -> list[str]:
    return [
        "crop_id",
        "audit_category",
        "source_audit_category",
        "human_label",
        "binary_true_vehicle",
        "focus_sv_score",
        "eqtext_sv_similarity",
        "positive_text_similarity",
        "max_negative_text_similarity",
        "negative_text_margin",
        "visual_text_consistency",
        "orientation_theta",
        "orientation_confidence",
        "fourier_phase_norm",
        "fft_low",
        "fft_mid",
        "fft_high",
        "fft_entropy",
        "padding_overlap_ratio",
        "degenerate_risk",
        *method_fields(),
        *idea_fields(),
        "tile_id",
        "angle",
        "image_path_zoom",
    ]


def _top_rate(rows: Sequence[Mapping[str, Any]], method_id: str,
              category: str, top_fraction: float = 0.20) -> float:
    if not rows:
        return 0.0
    count = max(1, int(math.ceil(len(rows) * top_fraction)))
    ordered = sorted(
        rows,
        key=lambda row: safe_float(row.get(method_id)),
        reverse=True)
    top = ordered[:count]
    return sum(row.get("audit_category") == category for row in top) / len(top)


def _true_retention(rows: Sequence[Mapping[str, Any]], method_id: str,
                    threshold: float = 0.5) -> float:
    true_rows = [
        row for row in rows
        if str(row.get("binary_true_vehicle")) in {"1", "1.0"}
        and safe_float(row.get("focus_sv_score")) >= threshold
    ]
    if not true_rows:
        return 1.0
    kept = sum(safe_float(row.get(method_id)) >= threshold for row in true_rows)
    return kept / len(true_rows)


def summarize_methods(rows: Sequence[Mapping[str, Any]],
                      baseline_auc: float = BASELINE_AUC) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method in METHOD_REGISTRY:
        method_id = method.method_id
        labeled = [
            row for row in rows
            if str(row.get("binary_true_vehicle")) in {"0", "1", "0.0", "1.0"}
        ]
        labels = [int(float(row["binary_true_vehicle"])) for row in labeled]
        scores = [safe_float(row.get(method_id)) for row in labeled]
        auc = roc_auc(labels, scores)
        values = [safe_float(row.get(method_id)) for row in rows]
        score_std = pstdev(values) if len(values) > 1 else 0.0
        collapse_flag = score_std < 1e-8
        degenerate_top20 = _top_rate(rows, method_id, "degenerate_large_sv_box")
        padding_top20 = _top_rate(rows, method_id, "padding_artifact")
        true_retention = _true_retention(rows, method_id)
        one_way_down_only = ""
        if method_id == "TF10_one_way_downweight":
            one_way_down_only = all(
                safe_float(row.get(method_id)) <= safe_float(row.get("focus_sv_score")) + 1e-12
                for row in rows)
        auc_delta = "" if auc is None else auc - baseline_auc
        rank_score = (
            -999.0 if auc is None else
            auc - 0.10 * degenerate_top20 - 0.10 * padding_top20
            - 0.05 * float(collapse_flag))
        reject_reasons: list[str] = []
        if auc is None:
            reject_reasons.append("missing_auc")
        elif auc - baseline_auc < 0.03:
            reject_reasons.append("auc_delta_below_0.03")
        if degenerate_top20 > 0.30:
            reject_reasons.append("degenerate_top20_above_0.30")
        if padding_top20 > 0.30:
            reject_reasons.append("padding_top20_above_0.30")
        if collapse_flag:
            reject_reasons.append("score_collapse")
        if method_id == "TF10_one_way_downweight" and one_way_down_only is not True:
            reject_reasons.append("tf10_not_one_way_down")
        safe_candidate = len(reject_reasons) == 0
        summary.append({
            "method_id": method_id,
            "name_zh": method.name_zh,
            "description_zh": method.description_zh,
            "auc": auc if auc is not None else "",
            "auc_delta_vs_tsafe": auc_delta,
            "rank_score": rank_score,
            "degenerate_top20_share": degenerate_top20,
            "padding_top20_share": padding_top20,
            "true_retention_proxy": true_retention,
            "score_std": score_std,
            "collapse_flag": collapse_flag,
            "one_way_down_only": one_way_down_only,
            "safe_candidate": safe_candidate,
            "reject_reason": "PASS_SAFE_CANDIDATE" if safe_candidate else ";".join(reject_reasons),
        })
    summary.sort(key=lambda row: safe_float(row["rank_score"], -999.0), reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def summarize_ideas(rows: Sequence[Mapping[str, Any]],
                    baseline_auc: float = BASELINE_AUC) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    labeled = [
        row for row in rows
        if str(row.get("binary_true_vehicle")) in {"0", "1", "0.0", "1.0"}
    ]
    for idea in IDEA_REGISTRY:
        idea_id = idea.idea_id
        labels = [int(float(row["binary_true_vehicle"])) for row in labeled]
        scores = [safe_float(row.get(idea_id)) for row in labeled]
        auc = roc_auc(labels, scores)
        values = [safe_float(row.get(idea_id)) for row in rows]
        score_std = pstdev(values) if len(values) > 1 else 0.0
        collapse_flag = score_std < 1e-8
        degenerate_top20 = _top_rate(rows, idea_id, "degenerate_large_sv_box")
        padding_top20 = _top_rate(rows, idea_id, "padding_artifact")
        true_retention = _true_retention(rows, idea_id)
        one_way_down_only: bool | str = ""
        if idea_id in {
                "I010", "I076", "I077", "I078", "I079", "I080",
                "I081", "I082", "I090", "I100",
        }:
            one_way_down_only = all(
                safe_float(row.get(idea_id)) <= safe_float(row.get("focus_sv_score")) + 1e-12
                for row in rows)
        auc_delta = "" if auc is None else auc - baseline_auc
        rank_score = (
            -999.0 if auc is None else
            auc - 0.10 * degenerate_top20 - 0.10 * padding_top20
            - 0.05 * float(collapse_flag))
        reject_reasons: list[str] = []
        if auc is None:
            reject_reasons.append("missing_auc")
        elif auc - baseline_auc < 0.03:
            reject_reasons.append("auc_delta_below_0.03")
        if degenerate_top20 > 0.30:
            reject_reasons.append("degenerate_top20_above_0.30")
        if padding_top20 > 0.30:
            reject_reasons.append("padding_top20_above_0.30")
        if true_retention < 0.50:
            reject_reasons.append("true_retention_below_0.50")
        if collapse_flag:
            reject_reasons.append("score_collapse")
        if one_way_down_only is False:
            reject_reasons.append("one_way_down_violation")
        safe_candidate = len(reject_reasons) == 0
        evidence_status = "PASS_NUMERIC_PROXY_EVIDENCE" if auc is not None else "NO_LABELED_AUC"
        summary.append({
            "idea_id": idea_id,
            "title_zh": idea.title_zh,
            "source_ids": idea.source_ids,
            "verification_zh": idea.verification_zh,
            "auc": auc if auc is not None else "",
            "auc_delta_vs_tsafe": auc_delta,
            "rank_score": rank_score,
            "degenerate_top20_share": degenerate_top20,
            "padding_top20_share": padding_top20,
            "true_retention_proxy": true_retention,
            "score_std": score_std,
            "collapse_flag": collapse_flag,
            "one_way_down_only": one_way_down_only,
            "safe_candidate": safe_candidate,
            "reject_reason": "PASS_SAFE_CANDIDATE" if safe_candidate else ";".join(reject_reasons),
            "sample_count": len(rows),
            "labeled_count": len(labeled),
            "evidence_status": evidence_status,
        })
    summary.sort(key=lambda row: safe_float(row["rank_score"], -999.0), reverse=True)
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
    return summary


def load_baseline_by_crop(path: Path) -> dict[str, dict[str, str]]:
    return {
        row.get("crop_id", ""): row
        for row in read_csv(path)
        if row.get("crop_id")
    }


def chinese_bool(value: Any) -> str:
    return "是" if bool(value) else "否"
