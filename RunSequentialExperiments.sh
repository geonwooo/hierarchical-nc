#!/bin/bash
# RunSequentialExperiments.sh — per-group sequential
# Usage: bash RunSequentialExperiments.sh [SEED]
set -e

SEED=${1:-0}
O="ddp False dp False mixed_precision False seed_num ${SEED}"
C="configs/cifar100/seq_base.yaml"
FC="configs/cifar100/ce_cifar100_resnet32.yaml"
G="groupings/seed${SEED}"
NC="nc_stats/seed${SEED}/seq"
mkdir -p ${NC}

echo "============================================"
echo " HNC Per-Group Sequential — Seed ${SEED}"
echo "============================================"

# Round 1: Core (4 GPU)
echo ">>> Round 1: Flat / Direct / Residual / ProbW"

FLAT_CKPT="output/cifar100/CE.CIFAR100.ResNet32.200epoch/seed$(printf '%03d' ${SEED})/models/best_model.pth"
if [ ! -f "${FLAT_CKPT}" ]; then
    python main/train.py --cfg ${FC} ${O} rank 0 proctitle flat &
    PID0=$!
else
    echo "  Flat — SKIPPED (exists)"
    PID0=""
fi

python main/train.py --cfg ${C} ${O} rank 1 \
    name "seq.Direct.R32" dataset.hier_type "true_seq_direct" \
    loss.lambda_coarse 1.0 proctitle seq_dir &
PID1=$!

python main/train.py --cfg ${C} ${O} rank 2 \
    name "seq.Residual.R32" dataset.hier_type "true_seq_residual" \
    loss.lambda_coarse 1.0 proctitle seq_res &
PID2=$!

python main/train.py --cfg ${C} ${O} rank 3 \
    name "seq.ProbW.R32" dataset.hier_type "true_seq_probw" \
    loss.lambda_coarse 1.0 proctitle seq_pw &
PID3=$!

[ -n "$PID0" ] && wait $PID0
wait $PID1 $PID2 $PID3
echo ">>> Round 1 DONE"

# Round 2: lambda sweep (4 GPU)
echo ">>> Round 2: lambda sweep (Direct)"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "seq.Direct-L03.R32" dataset.hier_type "true_seq_direct" \
    loss.lambda_coarse 0.3 proctitle l03 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "seq.Direct-L05.R32" dataset.hier_type "true_seq_direct" \
    loss.lambda_coarse 0.5 proctitle l05 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "seq.Direct-L20.R32" dataset.hier_type "true_seq_direct" \
    loss.lambda_coarse 2.0 proctitle l20 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "seq.Direct-L30.R32" dataset.hier_type "true_seq_direct" \
    loss.lambda_coarse 3.0 proctitle l30 &

wait
echo ">>> Round 2 DONE"

# Round 3: DualObj + Random + NoSG (4 GPU)
echo ">>> Round 3: DualObj / Random / NoSG"

python main/train.py --cfg ${C} ${O} rank 0 \
    name "seq.DualObj-S1.R32" dataset.hier_type "true_seq_direct" \
    loss.two_stage_mode "coarse_only" train.num_epochs 100 \
    proctitle dual_s1 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "seq.Direct-Rand.R32" dataset.hier_type "true_seq_direct" \
    dataset.random_hierarchy True loss.lambda_coarse 1.0 \
    proctitle dir_rand &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "seq.Residual-Rand.R32" dataset.hier_type "true_seq_residual" \
    dataset.random_hierarchy True loss.lambda_coarse 1.0 \
    proctitle res_rand &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "seq.ProbW-NoSG.R32" dataset.hier_type "true_seq_probw_nosg" \
    loss.lambda_coarse 1.0 proctitle pw_nosg &

wait

# DualObj Stage 2
S1_CKPT="output/cifar100/seq.DualObj-S1.R32/seed$(printf '%03d' ${SEED})/models/best_model.pth"
if [ ! -f "${S1_CKPT}" ]; then
    S1_CKPT=$(ls output/cifar100/seq.DualObj-S1.R32/seed$(printf '%03d' ${SEED})/models/epoch_*.pth 2>/dev/null | sort -V | tail -1)
fi
if [ -f "${S1_CKPT}" ]; then
    python main/train.py --cfg ${C} ${O} rank 0 \
        name "seq.DualObj-S2.R32" dataset.hier_type "true_seq_direct" \
        loss.two_stage_mode "joint" pretrained "${S1_CKPT}" \
        proctitle dual_s2
fi

echo ">>> Round 3 DONE"

# Round 4: Unsup grouping (2 GPU)
echo ">>> Round 4: Unsup grouping"
if [ -f "${G}/kmeans.json" ]; then
    python main/train.py --cfg ${C} ${O} rank 0 \
        name "seq.Direct-KM.R32" dataset.dataset "UnsupHierCIFAR100" \
        dataset.hier_type "true_seq_direct" \
        dataset.grouping_file "${G}/kmeans.json" \
        loss.lambda_coarse 1.0 proctitle dir_km &

    python main/train.py --cfg ${C} ${O} rank 1 \
        name "seq.Direct-Conf.R32" dataset.dataset "UnsupHierCIFAR100" \
        dataset.hier_type "true_seq_direct" \
        dataset.grouping_file "${G}/confusion.json" \
        loss.lambda_coarse 1.0 proctitle dir_conf &

    wait
else
    echo "  SKIP — groupings not found"
fi
echo ">>> Round 4 DONE"

# Results
echo ""
echo "============================================"
echo " RESULTS"
echo "============================================"
printf "%-35s %10s\n" "Model" "Best Acc"

if [ -f "${FLAT_CKPT}" ]; then
    FLAT_LOG=$(find "output/cifar100/CE.CIFAR100.ResNet32.200epoch/seed$(printf '%03d' ${SEED})" -name "*.log" 2>/dev/null | head -1)
    FLAT_ACC=$(grep "Best Acc" "${FLAT_LOG}" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "%-35s %10s\n" "Flat" "${FLAT_ACC}%"
fi

for d in output/cifar100/seq.*.R32; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    l=$(find "$d/seed$(printf '%03d' ${SEED})/logs" -name "*.log" 2>/dev/null | head -1)
    [ -z "$l" ] && continue
    acc=$(grep "Best Acc" "$l" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "%-35s %10s\n" "$n" "${acc:-?}%"
done
echo "============================================"
