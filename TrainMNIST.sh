#!/bin/bash
GPU_ID=${1}
DDP=${2}
DP=${3}
TITLE=${4}
CFG="configs/mnist/ce_mnist_lenet5.yaml"

if [ ${DDP} = "True" ]
then
    GPU_OPT="ddp True"
elif [ ${DP} = "True" ]
then
    GPU_OPT="ddp False dp True"
else
    GPU_OPT="ddp False dp False rank ${GPU_ID}"
fi

ARGS="
    --cfg ${CFG} 
    ${GPU_OPT}
    mixed_precision False
    proctitle ${TITLE}
    train.num_epochs 1
    train.trainer.type mixup
"


python main/train.py ${ARGS}

