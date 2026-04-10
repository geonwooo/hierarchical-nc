#!/bin/bash
# v3: 모든 모델이 100-way fine accuracy로 비교
# 6개 모델, 4 GPU, 2 라운드

SEED=${1:-0}
OPT="ddp False dp False mixed_precision False seed_num ${SEED}"

echo "========================================"
echo " v3 Pilot — 전부 fine_100 accuracy"
echo "========================================"

# Round 1: 4개
echo ">>> Round 1: A(Flat) / B(Sequential) / C(Factorized) / E(SeqRes)"

python main/train.py --cfg configs/cifar100/v3_flat.yaml        ${OPT} rank 0 proctitle v3_flat &
python main/train.py --cfg configs/cifar100/v3_sequential.yaml  ${OPT} rank 1 proctitle v3_seq &
python main/train.py --cfg configs/cifar100/v3_factorized.yaml  ${OPT} rank 2 proctitle v3_fact &
python main/train.py --cfg configs/cifar100/v3_seqres.yaml      ${OPT} rank 3 proctitle v3_seqres &
wait
echo ">>> Round 1 DONE"

# Round 2: 2개 (대조군)
echo ">>> Round 2: D(Fact+Rand) / F(Fact+Hard)"

python main/train.py --cfg configs/cifar100/v3_fact_rand.yaml   ${OPT} rank 0 proctitle v3_rand &
python main/train.py --cfg configs/cifar100/v3_fact_hard.yaml   ${OPT} rank 1 proctitle v3_hard &
wait
echo ">>> Round 2 DONE"

# 결과 확인
echo ""
echo "========================================"
echo " RESULTS (전부 fine_100 accuracy)"
echo "========================================"
for d in output/cifar100/v3.*/; do
  name=$(basename "$d")
  acc=$(grep "Best Acc" "$d/seed000/logs/"*.log 2>/dev/null | tail -1 | grep -oP 'Best Acc:\s*\K[0-9.]+')
  printf "%-28s %s%%\n" "$name" "$acc"
done
echo "========================================"
