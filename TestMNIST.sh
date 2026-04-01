#!/bin/bash
GPU_ID=${1}
DDP=${2}
DP=${3}
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
    pretrained ./output/mnist/CE.MNIST.LeNet5.20epoch/seed000/models/best_model.pth
    eval_mode True
    name FT.MNIST.LeNet5
"


python main/test.py ${ARGS}

