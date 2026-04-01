#!/bin/bash
# CompareCIFAR100.sh
# Trains and tests both CIFAR100 and HierCIFAR100 models, then prints a
# side-by-side accuracy comparison.
#
# Usage: bash CompareCIFAR100.sh <GPU_ID> <DDP> <DP> <TITLE>
#   GPU_ID : GPU index to use in single-GPU mode
#   DDP    : "True" to use DistributedDataParallel, "False" otherwise
#   DP     : "True" to use DataParallel,             "False" otherwise
#   TITLE  : proctitle prefix (suffixed with _CIFAR100 / _HierCIFAR100)

GPU_ID=${1}
DDP=${2}
DP=${3}
TITLE=${4}

# ---------- Train ----------
echo "========================================"
echo " [1/4] Training CIFAR100"
echo "========================================"
bash TrainCIFAR100.sh ${GPU_ID} ${DDP} ${DP} ${TITLE}_CIFAR100

echo "========================================"
echo " [2/4] Training HierCIFAR100"
echo "========================================"
bash TrainHierCIFAR100.sh ${GPU_ID} ${DDP} ${DP} ${TITLE}_HierCIFAR100

# ---------- Test ----------
echo "========================================"
echo " [3/4] Testing CIFAR100"
echo "========================================"
CIFAR100_LOG=$(bash TestCIFAR100.sh ${GPU_ID} ${DDP} ${DP} 2>&1)
echo "${CIFAR100_LOG}"
CIFAR100_ACC=$(echo "${CIFAR100_LOG}" | grep "Test Accuracy" | awk '{print $NF}')

echo "========================================"
echo " [4/4] Testing HierCIFAR100"
echo "========================================"
HIER_LOG=$(bash TestHierCIFAR100.sh ${GPU_ID} ${DDP} ${DP} 2>&1)
echo "${HIER_LOG}"
HIER_ACC=$(echo "${HIER_LOG}" | grep "Test Accuracy" | awk '{print $NF}')

# ---------- Comparison ----------
echo ""
echo "=================================================="
echo "              Comparison Results                  "
echo "=================================================="
printf "%-38s  %s\n" "Model"                           "Test Acc"
printf "%-38s  %s\n" "------"                          "--------"
printf "%-38s  %s\n" "CIFAR100    (flat,  100-class)"  "${CIFAR100_ACC}"
printf "%-38s  %s\n" "HierCIFAR100 (pred_1 ∩ pred_2)" "${HIER_ACC}"
echo "=================================================="
