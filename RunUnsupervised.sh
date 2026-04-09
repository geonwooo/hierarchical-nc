#!/bin/bash
# RunUnsupervised.sh — Phase 1
# 실행: bash RunUnsupervised.sh <GPU_ID> <SEED>

GPU_ID=${1:-0}
SEED=${2:-0}
OPT="ddp False dp False mixed_precision False seed_num ${SEED} rank ${GPU_ID}"

FLAT_CFG="configs/cifar100/ce_cifar100_resnet32.yaml"
FLAT_NAME="CE.CIFAR100.ResNet32.200epoch"
FLAT_CKPT="output/cifar100/${FLAT_NAME}/seed$(printf '%03d' ${SEED})/models/best_model.pth"
UNSUP_CFG="configs/cifar100/v3_unsup_base.yaml"
GRP="groupings/seed${SEED}"
NC="nc_stats/seed${SEED}"
mkdir -p ${GRP} ${NC}

# Stage 1: Flat + discover
if [ ! -f "${FLAT_CKPT}" ]; then
    echo "[1] Flat baseline..."
    python main/train.py --cfg ${FLAT_CFG} ${OPT} proctitle flat
fi

echo "[Discover] hierarchies..."
for M in kmeans confusion random; do
    python tools/discover_hierarchy.py --method ${M} --checkpoint ${FLAT_CKPT} --cfg ${FLAT_CFG} --num-groups 20 --output ${GRP}/${M}.json --seed ${SEED} --rank ${GPU_ID}
done

# Stage 2: Train 5 models
for PAIR in "UnsupKmeans:kmeans" "UnsupRandom:random" "UnsupConfusion:confusion"; do
    NAME="${PAIR%%:*}"; FILE="${PAIR##*:}"
    echo "[Train] ${NAME}..."
    python main/train.py --cfg ${UNSUP_CFG} ${OPT} name "v3.${NAME}.R32" dataset.grouping_file "${GRP}/${FILE}.json" proctitle ${NAME}
done

echo "[Train] UnsupKmeans+LamWarm..."
python main/train.py --cfg ${UNSUP_CFG} ${OPT} name "v3.UnsupKmeans-LW.R32" dataset.grouping_file "${GRP}/kmeans.json" loss.lambda_warmup True proctitle km_lw

echo "[Train] Supervised (oracle)..."
python main/train.py --cfg ${UNSUP_CFG} ${OPT} name "v3.Supervised.R32" dataset.grouping_file "none" proctitle sup

# NC stats
echo "[NC] Collecting stats..."
python tools/collect_nc_stats_cifar.py --cfg ${FLAT_CFG} --checkpoint ${FLAT_CKPT} --output ${NC}/flat.json --rank ${GPU_ID}
for PAIR in "UnsupKmeans:kmeans" "UnsupRandom:random" "UnsupConfusion:confusion" "UnsupKmeans-LW:kmeans"; do
    NAME="${PAIR%%:*}"; FILE="${PAIR##*:}"
    CKPT="output/cifar100/v3.${NAME}.R32/seed$(printf '%03d' ${SEED})/models/best_model.pth"
    [ -f "${CKPT}" ] && python tools/collect_nc_stats_cifar.py --cfg ${UNSUP_CFG} --checkpoint ${CKPT} --output ${NC}/${NAME}.json --grouping "${GRP}/${FILE}.json" --rank ${GPU_ID}
done

# Results
echo ""
echo "========== RESULTS =========="
printf "%-30s %8s\n" "Model" "Acc%"
for d in output/cifar100/v3.*.R32 output/cifar100/${FLAT_NAME}; do
    [ -d "$d" ] || continue
    log=$(ls "$d/seed$(printf '%03d' ${SEED})/logs/"*.log 2>/dev/null | head -1)
    [ -z "$log" ] && continue
    acc=$(grep "Best Acc" "$log" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "%-30s %8s\n" "$(basename $d)" "${acc:-?}%"
done
echo "============================="
