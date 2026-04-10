#!/bin/bash
# RunPilotCompare.sh
# Trains 5 models on CIFAR-100 for NC pilot comparison.
# Usage: bash RunPilotCompare.sh <GPU_ID> <SEED>
#
# Models:
#   1. Flat baseline     (100-way CE)
#   2. Sequential        (선배님 original: 20→sg(Wp)→5)
#   3. Sequential+Rand   (random hierarchy 대조군)
#   4. SeqResidual       (20→h+α·sg(Wp)→5)
#   5. Factorized        (coarse(20) + per-group fine(5) → 100-way)
#   6. Factorized+Rand   (random hierarchy 대조군)

GPU_ID=${1:-0}
SEED=${2:-0}
GPU_OPT="ddp False dp False rank ${GPU_ID} mixed_precision False"

echo "============================================"
echo " NC Pilot Comparison — GPU ${GPU_ID}, Seed ${SEED}"
echo "============================================"

# 1. Flat baseline
echo ""
echo "[1/6] Flat baseline (100-way)"
python main/train.py \
    --cfg configs/cifar100/ce_cifar100_vgg11.yaml \
    ${GPU_OPT} seed_num ${SEED} \
    proctitle pilot_flat

# 2. Sequential (선배님 original)
echo ""
echo "[2/6] Sequential (선배님, semantic hierarchy)"
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11.yaml \
    ${GPU_OPT} seed_num ${SEED} \
    proctitle pilot_seq

# 3. Sequential + Random hierarchy (대조군)
echo ""
echo "[3/6] Sequential (random hierarchy, 대조군)"
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11.yaml \
    ${GPU_OPT} seed_num ${SEED} \
    dataset.random_hierarchy True \
    name CE.HierCIFAR100-Rand.VGG11.200epoch \
    proctitle pilot_seq_rand

# 4. Sequential Residual (h + α·sg(Wp))
echo ""
echo "[4/6] Sequential Residual (h + α·sg(Wp))"
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11_seqres.yaml \
    ${GPU_OPT} seed_num ${SEED} \
    proctitle pilot_seqres

# 5. Factorized (건우님, semantic hierarchy)
echo ""
echo "[5/6] Factorized (coarse+fine → 100-way, semantic)"
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11_factorized.yaml \
    ${GPU_OPT} seed_num ${SEED} \
    proctitle pilot_fact

# 6. Factorized + Random hierarchy (대조군)
echo ""
echo "[6/6] Factorized (random hierarchy, 대조군)"
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11_factorized_rand.yaml \
    ${GPU_OPT} seed_num ${SEED} \
    proctitle pilot_fact_rand

echo ""
echo "============================================"
echo " All 6 models trained. Check ./output/cifar100/"
echo "============================================"
echo ""
echo "Expected results to compare:"
echo "  Flat:           fine_100 acc  (~67%)"
echo "  Sequential:     joint_20+5   (선배님 ~78%)"
echo "  Seq+Rand:       joint_20+5   (random 대조군, 낮을 것)"
echo "  SeqResidual:    joint_20+5   (residual fix 효과 확인)"
echo "  Factorized:     fine_100 acc  (우리 방식)"
echo "  Fact+Rand:      fine_100 acc  (random 대조군)"
echo ""
echo "NOTE: Sequential의 joint_20+5와 Factorized의 fine_100은"
echo "      다른 metric이므로 직접 비교 불가. 둘 다 보고해야 함."
