#!/bin/bash
# RunFullSweep.sh — 32 experiments, 8 rounds × 4 GPU
# 각 round ~40min, 총 ~5-6시간
# Usage: bash RunFullSweep.sh [SEED]
set -e

SEED=${1:-0}
O="ddp False dp False mixed_precision False seed_num ${SEED}"
C="configs/cifar100/seq_base.yaml"
T="true_seq_direct"

echo "============================================"
echo " HNC Full Sweep — 32 experiments, Seed ${SEED}"
echo " Baseline: Direct 69.99% (coarse 81.34%, fine_oracle 83.03%)"
echo " Target: 71%+ (close gap with Flat 71.52%)"
echo "============================================"

# ============================================================
# R1: Fine MLP — fine_oracle 83% 개선 목표
# ============================================================
echo ">>> R1: Fine MLP capacity sweep"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.MLP-H16.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 16 \
    proctitle mlp16 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.MLP-H32.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    proctitle mlp32 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.MLP-H64.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 64 \
    proctitle mlp64 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.MLP-H128.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 128 \
    proctitle mlp128 &

wait
echo ">>> R1 DONE"

# ============================================================
# R2: Scheduled Sampling + Aux 100-way loss
# ============================================================
echo ">>> R2: SS + Aux100"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.SS.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.scheduled_sampling True \
    proctitle ss &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.Aux03.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.aux100_weight 0.3 \
    proctitle aux03 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.Aux05.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.aux100_weight 0.5 \
    proctitle aux05 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.Aux10.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.aux100_weight 1.0 \
    proctitle aux10 &

wait
echo ">>> R2 DONE"

# ============================================================
# R3: Cosine classifier + Fine label smooth
# ============================================================
echo ">>> R3: Cosine + FineLabelSmooth"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.Cos16.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 16.0 \
    proctitle cos16 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.Cos32.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 \
    proctitle cos32 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.FLS01.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_label_smooth 0.1 \
    proctitle fls01 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.FLS02.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_label_smooth 0.2 \
    proctitle fls02 &

wait
echo ">>> R3 DONE"

# ============================================================
# R4: MLP + single combo
# ============================================================
echo ">>> R4: MLP32 + combos"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.MLP32-SS.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True \
    proctitle m32ss &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.MLP32-Aux05.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.aux100_weight 0.5 \
    proctitle m32a05 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.MLP32-Cos.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.cosine_classifier True loss.cosine_scale 16.0 \
    proctitle m32cos &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.MLP32-FLS.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.fine_label_smooth 0.1 \
    proctitle m32fls &

wait
echo ">>> R4 DONE"

# ============================================================
# R5: Triple combos
# ============================================================
echo ">>> R5: Triple combos"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.MLP32-SS-Aux05.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.scheduled_sampling True loss.aux100_weight 0.5 \
    proctitle m32ssa &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.MLP32-Cos-Aux05.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.cosine_classifier True loss.cosine_scale 16.0 loss.aux100_weight 0.5 \
    proctitle m32ca &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.MLP32-SS-FLS.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.scheduled_sampling True loss.fine_label_smooth 0.1 \
    proctitle m32sf &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.MLP32-Aux05-FLS.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.aux100_weight 0.5 loss.fine_label_smooth 0.1 \
    proctitle m32af &

wait
echo ">>> R5 DONE"

# ============================================================
# R6: Residual variants
# ============================================================
echo ">>> R6: Residual + best combos"
TR="true_seq_residual"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.Res-MLP32.R32" dataset.hier_type ${TR} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    proctitle rm32 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.Res-MLP32-Aux05.R32" dataset.hier_type ${TR} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.aux100_weight 0.5 \
    proctitle rm32a &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.Res-MLP32-SS.R32" dataset.hier_type ${TR} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True \
    proctitle rm32s &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.Res-MLP32-SS-Aux05.R32" dataset.hier_type ${TR} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.scheduled_sampling True loss.aux100_weight 0.5 \
    proctitle rm32sa &

wait
echo ">>> R6 DONE"

# ============================================================
# R7: Coarse improvements + λ with MLP
# ============================================================
echo ">>> R7: Coarse improve + λ"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.SB01-MLP32.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.soft_beta 0.1 \
    proctitle sb01 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.SB02-MLP32.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.soft_beta 0.2 \
    proctitle sb02 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.L05-MLP32.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 0.5 loss.fine_hidden 32 \
    proctitle l05m &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.L20-MLP32.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 2.0 loss.fine_hidden 32 \
    proctitle l20m &

wait
echo ">>> R7 DONE"

# ============================================================
# R8: Full combo + coarse soft_beta
# ============================================================
echo ">>> R8: Full combos"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "v4.MLP32-SS-Aux05-FLS.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.scheduled_sampling True loss.aux100_weight 0.5 \
    loss.fine_label_smooth 0.1 \
    proctitle full1 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "v4.MLP32-SS-Aux05-SB01.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.scheduled_sampling True loss.aux100_weight 0.5 \
    loss.soft_beta 0.1 \
    proctitle full2 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "v4.MLP64-SS-Aux05.R32" dataset.hier_type ${T} \
    loss.lambda_coarse 1.0 loss.fine_hidden 64 \
    loss.scheduled_sampling True loss.aux100_weight 0.5 \
    proctitle full3 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "v4.Res-MLP32-SS-Aux05-FLS.R32" dataset.hier_type ${TR} \
    loss.lambda_coarse 1.0 loss.fine_hidden 32 \
    loss.scheduled_sampling True loss.aux100_weight 0.5 \
    loss.fine_label_smooth 0.1 \
    proctitle full4 &

wait
echo ">>> R8 DONE"

# ============================================================
# RESULTS
# ============================================================
echo ""
echo "============================================"
echo " ALL RESULTS — sorted by accuracy"
echo "============================================"
printf "%-40s %8s %8s %8s\n" "Model" "Joint%" "Coarse%" "FineOr%"

for d in output/cifar100/v4.*.R32; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    l=$(find "$d/seed$(printf '%03d' ${SEED})/logs" -name "*.log" 2>/dev/null | head -1)
    [ -z "$l" ] && continue
    joint=$(grep "Best Acc" "$l" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    coarse=$(grep "coarse:" "$l" 2>/dev/null | tail -1 | grep -oP 'coarse:[0-9.]+' | head -1 | grep -oP '[0-9.]+')
    fine_or=$(grep "fine_oracle:" "$l" 2>/dev/null | tail -1 | grep -oP 'fine_oracle:[0-9.]+' | head -1 | grep -oP '[0-9.]+')
    printf "%-40s %8s %8s %8s\n" "$n" "${joint:-?}%" "${coarse:-?}%" "${fine_or:-?}%"
done | sort -t'%' -k2 -rn

echo ""
echo "Baselines:"
echo "  Flat:     71.52%"
echo "  Direct:   69.99% (coarse 81.34%, fine_oracle 83.03%)"
echo "  Residual: 69.77% (coarse 81.59%, fine_oracle 82.64%)"
echo "============================================"
