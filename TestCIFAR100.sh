#!/bin/bash
GPU_ID=${1}
DDP=${2}
DP=${3}
CFG="configs/cifar100/ce_cifar100_vgg11.yaml"

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
    pretrained ./output/cifar100/CE.CIFAR100.VGG11.200epoch/seed000/models/best_model.pth
    eval_mode True
    name FT.CIFAR100.VGG11
"


python main/test.py ${ARGS}
