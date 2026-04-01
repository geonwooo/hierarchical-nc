#!/bin/bash
GPU_ID=${1}
RAND=${2}
TITLE=${3}

#SEEDS=(
#     28  29  32  35  47
#     73 217 251 254 308
#    370 392 402 561 662
#    833 900 923 925 995
#)
SEEDS=(28  29  32  35  47)
#SEEDS=(73 402 995)
#SEEDS=(1 2)

title="${TITLE}_rand${RAND}_seed${SEED}"
for SEED in ${SEEDS[@]}
do
    ./TrainHierCIFAR100.sh ${GPU_ID} ${RAND} ${title} ${SEED}
done

