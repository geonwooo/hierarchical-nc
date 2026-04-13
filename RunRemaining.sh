#!/bin/bash
# 현재 돌고있는 4개 끝나면 실행
# Usage: bash RunRemaining.sh
set -e
O="ddp False dp False mixed_precision False seed_num 0"
C="configs/cifar100/seq_base.yaml"
T="true_seq_direct"
TR="true_seq_residual"

# Batch 1: MLP sweep + MLP32-Cos (GPU 0-3)
echo ">>> Batch 1"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.MLP-H16.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 16 proctitle mlp16 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.MLP-H64.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 64 proctitle mlp64 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.MLP-H128.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 128 proctitle mlp128 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.MLP32-Cos.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.cosine_classifier True loss.cosine_scale 16.0 proctitle m32cos &
wait; echo ">>> Batch 1 DONE"

# Batch 2: MLP32-FLS + Triple combos
echo ">>> Batch 2"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.MLP32-FLS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.fine_label_smooth 0.1 proctitle m32fls &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.MLP32-SS-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle m32ssa &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.MLP32-Cos-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.cosine_classifier True loss.cosine_scale 16.0 loss.aux100_weight 0.5 proctitle m32ca &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.MLP32-Aux05-FLS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.aux100_weight 0.5 loss.fine_label_smooth 0.1 proctitle m32af &
wait; echo ">>> Batch 2 DONE"

# Batch 3: Residual
echo ">>> Batch 3"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.Res-MLP32.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 proctitle rm32 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.Res-MLP32-Aux05.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.aux100_weight 0.5 proctitle rm32a &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.Res-MLP32-SS.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True proctitle rm32s &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.Res-MLP32-SS-Aux05.R32" dataset.hier_type ${TR} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle rm32sa &
wait; echo ">>> Batch 3 DONE"

# Batch 4: Coarse improve + λ
echo ">>> Batch 4"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.SB01-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.soft_beta 0.1 proctitle sb01 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.SB02-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.soft_beta 0.2 proctitle sb02 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.L05-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 0.5 loss.fine_hidden 32 proctitle l05m &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.L20-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 2.0 loss.fine_hidden 32 proctitle l20m &
wait; echo ">>> Batch 4 DONE"

# Batch 5: FiLM + MLP
echo ">>> Batch 5"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.FiLM-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 proctitle filmm &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.FiLM-MLP32-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 loss.aux100_weight 0.5 proctitle filmma &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.FiLM-MLP32-SS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 loss.scheduled_sampling True proctitle filmms &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.ETFc.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True proctitle etfc &
wait; echo ">>> Batch 5 DONE"

# Batch 6: ETF + Joint 100-way
echo ">>> Batch 6"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.ETFc-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True loss.fine_hidden 32 proctitle etfcm &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.ETFc-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.etf_coarse True loss.aux100_weight 0.5 proctitle etfca &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.J100.R32" dataset.hier_type ${T} loss.joint_100way True proctitle j100 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.J100-MLP32.R32" dataset.hier_type ${T} loss.joint_100way True loss.fine_hidden 32 proctitle j100m &
wait; echo ">>> Batch 6 DONE"

# Batch 7: Full combos
echo ">>> Batch 7"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.MLP32-SS-Aux05-FLS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 loss.fine_label_smooth 0.1 proctitle full1 &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.MLP64-SS-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.fine_hidden 64 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle full3 &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.FiLM-MLP32-SS-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.use_film True loss.fine_hidden 32 loss.scheduled_sampling True loss.aux100_weight 0.5 proctitle ult1 &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.Cos32-MLP32.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 loss.fine_hidden 32 proctitle c32m &
wait; echo ">>> Batch 7 DONE"

# Batch 8: Cos32 combos (best single = 70.60%)
echo ">>> Batch 8"
python main/train.py --cfg ${C} ${O} rank 0 name "v4.Cos32-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 loss.aux100_weight 0.5 proctitle c32a &
python main/train.py --cfg ${C} ${O} rank 1 name "v4.Cos32-SS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 loss.scheduled_sampling True proctitle c32s &
python main/train.py --cfg ${C} ${O} rank 2 name "v4.Cos32-MLP32-Aux05.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 loss.fine_hidden 32 loss.aux100_weight 0.5 proctitle c32ma &
python main/train.py --cfg ${C} ${O} rank 3 name "v4.Cos32-MLP32-SS.R32" dataset.hier_type ${T} loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 loss.fine_hidden 32 loss.scheduled_sampling True proctitle c32ms &
wait; echo ">>> Batch 8 DONE"

# Results
echo ""
echo "============================================"
printf "%-42s %8s\n" "Model" "Best%"
printf "%-42s %8s\n" "Flat" "71.52%"
for d in output/cifar100/v4.*.R32; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    l=$(find "$d" -name "*.log" 2>/dev/null | head -1)
    [ -z "$l" ] && continue
    acc=$(grep "Best Acc" "$l" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    [ -z "$acc" ] && continue
    printf "%-42s %8s\n" "$n" "${acc}%"
done | sort -t'%' -k2 -rn
echo "============================================"
