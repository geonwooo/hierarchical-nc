#!/bin/bash
# RunPilotCompare_4GPU.sh
# 4 GPU 병렬 실행: 6개 모델을 2 라운드로 나눠서 돌림
# Usage: bash RunPilotCompare_4GPU.sh [SEED]

SEED=${1:-0}
COMMON="ddp False dp False mixed_precision False seed_num ${SEED}"

echo "============================================"
echo " NC Pilot — 4 GPU parallel, Seed ${SEED}"
echo " Round 1: 4 models (GPU 0-3)"
echo " Round 2: 2 models (GPU 0-1)"
echo "============================================"

# ===== Round 1: 4개 동시 =====
echo ""
echo ">>> Round 1 START (4 models parallel)"

# GPU 0: Flat baseline
python main/train.py \
    --cfg configs/cifar100/ce_cifar100_vgg11.yaml \
    ${COMMON} rank 0 proctitle pilot_flat &
PID0=$!

# GPU 1: Sequential (선배님 original)
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11.yaml \
    ${COMMON} rank 1 proctitle pilot_seq &
PID1=$!

# GPU 2: Factorized (건우님, semantic)
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11_factorized.yaml \
    ${COMMON} rank 2 proctitle pilot_fact &
PID2=$!

# GPU 3: Sequential Residual
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11_seqres.yaml \
    ${COMMON} rank 3 proctitle pilot_seqres &
PID3=$!

echo "  GPU 0: Flat (PID $PID0)"
echo "  GPU 1: Sequential (PID $PID1)"
echo "  GPU 2: Factorized (PID $PID2)"
echo "  GPU 3: SeqResidual (PID $PID3)"
echo "  Waiting..."

wait $PID0 $PID1 $PID2 $PID3
echo ">>> Round 1 DONE"

# ===== Round 2: 2개 동시 (대조군) =====
echo ""
echo ">>> Round 2 START (2 models parallel, 대조군)"

# GPU 0: Sequential + Random
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11.yaml \
    ${COMMON} rank 0 \
    dataset.random_hierarchy True \
    name CE.HierCIFAR100-Rand.VGG11.200epoch \
    proctitle pilot_seq_rand &
PID0=$!

# GPU 1: Factorized + Random
python main/train.py \
    --cfg configs/cifar100/ce_hiercifar100_vgg11_factorized_rand.yaml \
    ${COMMON} rank 1 proctitle pilot_fact_rand &
PID1=$!

echo "  GPU 0: Seq+Rand (PID $PID0)"
echo "  GPU 1: Fact+Rand (PID $PID1)"
echo "  Waiting..."

wait $PID0 $PID1
echo ">>> Round 2 DONE"

# ===== 결과 요약 =====
echo ""
echo "============================================"
echo " ALL DONE — Results in ./output/cifar100/"
echo "============================================"
echo ""
echo "  CE.CIFAR100.VGG11.200epoch                 → Flat baseline"
echo "  CE.HierCIFAR100.VGG11.200epoch             → Sequential "
echo "  CE.HierCIFAR100-Rand.VGG11.200epoch        → Sequential random"
echo "  CE.HierCIFAR100.SeqRes.VGG11.200epoch      → Sequential residual"
echo "  CE.HierCIFAR100.Factorized.VGG11.200epoch  → Factorized "
echo "  CE.HierCIFAR100.Factorized-Rand.VGG11.200epoch → Factorized random"