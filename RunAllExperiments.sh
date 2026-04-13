#!/bin/bash
# RunAllExperiments.sh — 48 experiments, 12 rounds × 4 GPU
# ~8시간 (미팅까지 16시간이니 충분)
# Usage: bash RunAllExperiments.sh [SEED]
set -e

SEED=${1:-0}
O="ddp False dp False mixed_precision False seed_num ${SEED}"
C="configs/cifar100/seq_base.yaml"
T="true_seq_direct"
TR="true_seq_residual"

echo "============================================"
echo " HNC Full — 48 experiments, Seed ${SEED}"
echo " Baseline: Direct 69.99%, Flat 71.52%"
echo "============================================"

# ============================================================
# PHASE 1: Fine capacity, Aux, SS, Cosine, Label Smooth, Combos
# ============================================================

# R1: Fine MLP
echo ">>> R1: Fine MLP"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.MLP-H16.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 16 proctitle mlp16 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.MLP-H32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 proctitle mlp32 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.MLP-H64.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 64 proctitle mlp64 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.MLP-H128.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 128 proctitle mlp128 &
wait; echo ">>> R1 DONE"

# R2: SS + Aux100
echo ">>> R2: SS + Aux100"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.SS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.scheduled_sampling True proctitle ss &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.Aux03.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.aux100_weight 0.3 proctitle aux03 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.aux100_weight 0.5 proctitle aux05 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.Aux10.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.aux100_weight 1.0 proctitle aux10 &
wait; echo ">>> R2 DONE"

# R3: Cosine + Fine label smooth
echo ">>> R3: Cosine + FLS"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.Cos16.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 16.0 proctitle cos16 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.Cos32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 proctitle cos32 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.FLS01.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_label_smooth 0.1 proctitle fls01 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.FLS02.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_label_smooth 0.2 proctitle fls02 &
wait; echo ">>> R3 DONE"

# R4: MLP32 + single combo
echo ">>> R4: MLP32 combos"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.MLP32-SS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True proctitle m32ss &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.MLP32-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.aux100_weight 0.5 proctitle m32a05 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.MLP32-Cos.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.cosine_classifier True loss.cosine_scale 16.0 proctitle m32cos &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.MLP32-FLS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.fine_label_smooth 0.1 proctitle m32fls &
wait; echo ">>> R4 DONE"

# R5: Triple combos
echo ">>> R5: Triple combos"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.MLP32-SS-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle m32ssa &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.MLP32-Cos-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.cosine_classifier True loss.cosine_scale 16.0 loss.aux100_weight 0.5 proctitle m32ca &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.MLP32-SS-FLS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.fine_label_smooth 0.1 proctitle m32sf &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.MLP32-Aux05-FLS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.aux100_weight 0.5 loss.fine_label_smooth 0.1 proctitle m32af &
wait; echo ">>> R5 DONE"

# R6: Residual variants
echo ">>> R6: Residual"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.Res-MLP32.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 proctitle rm32 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.Res-MLP32-Aux05.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.aux100_weight 0.5 proctitle rm32a &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.Res-MLP32-SS.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True proctitle rm32s &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.Res-MLP32-SS-Aux05.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle rm32sa &
wait; echo ">>> R6 DONE"

# R7: Coarse improvements
echo ">>> R7: Coarse improve"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.SB01-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.soft_beta 0.1 proctitle sb01 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.SB02-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.soft_beta 0.2 proctitle sb02 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.L05-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 0.5 loss.fine_hidden 32 proctitle l05m &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.L20-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 2.0 loss.fine_hidden 32 proctitle l20m &
wait; echo ">>> R7 DONE"

# R8: Full combos
echo ">>> R8: Full combos"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.MLP32-SS-Aux05-FLS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 loss.fine_label_smooth 0.1 proctitle full1 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.MLP32-SS-Aux05-SB01.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 loss.soft_beta 0.1 proctitle full2 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.MLP64-SS-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 64 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle full3 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.Res-MLP32-SS-Aux05-FLS.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 loss.fine_label_smooth 0.1 proctitle full4 &
wait; echo ">>> R8 DONE"

# ============================================================
# PHASE 2: FiLM, ETF, Joint 100-way
# ============================================================

# R9: FiLM
echo ">>> R9: FiLM"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.FiLM.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True proctitle film &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.FiLM-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 proctitle filmm &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.FiLM-MLP32-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 loss.aux100_weight 0.5 proctitle filmma &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.FiLM-MLP32-SS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 loss.scheduled_sampling True proctitle filmms &
wait; echo ">>> R9 DONE"

# R10: ETF
echo ">>> R10: ETF"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.ETFc.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True proctitle etfc &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.ETFcf.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True loss.etf_fine True proctitle etfcf &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.ETFc-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True loss.fine_hidden 32 proctitle etfcm &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.ETFc-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True loss.aux100_weight 0.5 proctitle etfca &
wait; echo ">>> R10 DONE"

# R11: Joint 100-way loss
echo ">>> R11: Joint 100-way"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.J100.R32" dataset.hier_type ${T} loss.joint_100way True proctitle j100 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.J100-MLP32.R32" dataset.hier_type ${T} loss.joint_100way True loss.fine_hidden 32 proctitle j100m &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.J100-FiLM.R32" dataset.hier_type ${T} loss.joint_100way True loss.use_film True proctitle j100f &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.J100-FiLM-MLP32.R32" dataset.hier_type ${T} loss.joint_100way True loss.use_film True loss.fine_hidden 32 proctitle j100fm &
wait; echo ">>> R11 DONE"

# R12: Best combos from Phase 1+2
echo ">>> R12: Ultimate combos"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.FiLM-MLP32-SS-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle ult1 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.Res-FiLM-MLP32.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 proctitle ult2 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.Res-FiLM-MLP32-Aux05.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 loss.aux100_weight 0.5 proctitle ult3 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.ETFc-FiLM-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True loss.use_film True loss.fine_hidden 32 proctitle ult4 &
wait; echo ">>> R12 DONE"

# ============================================================
# RESULTS
# ============================================================
echo ""
echo "============================================"
echo " ALL RESULTS — sorted by accuracy"
echo "============================================"
printf "%-42s %8s %8s %8s\n" "Model" "Joint%" "Coarse%" "FineOr%"
printf "%-42s %8s %8s %8s\n" "Flat" "71.52%" "-" "-"

for d in output/cifar100/v4.*.R32; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    l=$(find "$d/seed$(printf '%03d' ${SEED})/logs" -name "*.log" 2>/dev/null | head -1)
    [ -z "$l" ] && continue
    joint=$(grep "Best Acc" "$l" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    coarse=$(grep "coarse:" "$l" 2>/dev/null | tail -1 | grep -oP 'coarse:[0-9.]+' | head -1 | grep -oP '[0-9.]+')
    fine_or=$(grep "fine_oracle:" "$l" 2>/dev/null | tail -1 | grep -oP 'fine_oracle:[0-9.]+' | head -1 | grep -oP '[0-9.]+')
    printf "%-42s %8s %8s %8s\n" "$n" "${joint:-?}%" "${coarse:-?}%" "${fine_or:-?}%"
done | sort -t'%' -k2 -rn

echo "============================================"
