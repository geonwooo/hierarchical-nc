#!/bin/bash
GPU_ID=${1}
DDP="False"
DP="False"
RAND=${2}  # {True, False}
TITLE=${3}
SEED=${4}
CFG="configs/cifar100/ce_hiercifar100_resnet32.yaml"

if [ ${DDP} = "True" ]
then
    GPU_OPT="ddp True"
elif [ ${DP} = "True" ]
then
    GPU_OPT="ddp False dp True"
else
    GPU_OPT="ddp False dp False rank ${GPU_ID}"
fi

NAME="CE.HierCIFAR100-${RAND}.ResNet32.200epoch"
ARGS="
    --cfg ${CFG}
    ${GPU_OPT}
    name ${NAME}
    seed_num ${SEED}
    mixed_precision False
    dataset.random_hierarchy ${RAND}
    proctitle ${TITLE}
"


python main/train.py ${ARGS}
