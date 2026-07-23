#!/bin/bash
# RunHNC_StructureB.sh — Structure B experiments
# h^ℓ = genuine intermediate layer, h^L = backbone last → FC
set -e
cd ~/hierarchical-nc
S="python tools/hnc_train.py"
SEED=0

echo "========================================"
echo " HNC Structure B Validation"
echo " h^ℓ = intermediate, h^L = last → FC"
echo " $(date)"
echo "========================================"

# ═══════════════════════════════════════════
# Phase 1: CIFAR-100 Case (i) — 최우선
# ═══════════════════════════════════════════
echo ">>> Phase 1: CIFAR-100 Case (i)"
echo "    in_ch=64, h^ℓ=layer2(128), h^L=FC(layer3(256)→64)"
$S --dataset cifar100 --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset cifar100 --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar100 --case i --mode random_concat --rank 2 --seed $SEED &
$S --dataset cifar100 --case i --mode inter_only --rank 3 --seed $SEED &
wait
echo ">>> Phase 1 DONE $(date)"

# ═══════════════════════════════════════════
# Phase 2: CIFAR-100 Case (ii) + (iii)
# ═══════════════════════════════════════════
echo ">>> Phase 2a: CIFAR-100 Case (ii)"
$S --dataset cifar100 --case ii --mode baseline   --rank 0 --seed $SEED &
$S --dataset cifar100 --case ii --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar100 --case ii --mode random_concat   --rank 2 --seed $SEED &
$S --dataset cifar100 --case ii --mode inter_only --rank 3 --seed $SEED &
wait

echo ">>> Phase 2b: CIFAR-100 Case (iii)"
$S --dataset cifar100 --case iii --mode baseline   --rank 0 --seed $SEED &
$S --dataset cifar100 --case iii --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar100 --case iii --mode random_concat   --rank 2 --seed $SEED &
$S --dataset cifar100 --case iii --mode inter_only --rank 3 --seed $SEED &
wait
echo ">>> Phase 2 DONE $(date)"

# ═══════════════════════════════════════════
# Phase 3: Small datasets
# ═══════════════════════════════════════════
echo ">>> Phase 3: MNIST + CIFAR-10 + SVHN"
$S --dataset mnist   --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset mnist   --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset cifar10 --case i --mode baseline   --rank 2 --seed $SEED &
$S --dataset cifar10 --case i --mode concat     --rank 3 --seed $SEED &
wait

$S --dataset mnist   --case i --mode random_concat   --rank 0 --seed $SEED &
$S --dataset mnist   --case i --mode inter_only --rank 1 --seed $SEED &
$S --dataset cifar10 --case i --mode random_concat   --rank 2 --seed $SEED &
$S --dataset cifar10 --case i --mode inter_only --rank 3 --seed $SEED &
wait

$S --dataset svhn --case i --mode baseline   --rank 0 --seed $SEED &
$S --dataset svhn --case i --mode concat     --rank 1 --seed $SEED &
$S --dataset svhn --case i --mode random_concat   --rank 2 --seed $SEED &
$S --dataset svhn --case i --mode inter_only --rank 3 --seed $SEED &
wait
echo ">>> Phase 3 DONE $(date)"

# ═══════════════════════════════════════════
# Phase 4: Medium (Flowers, Food) — ResNet18
# ═══════════════════════════════════════════
echo ">>> Phase 4: Flowers + Food (ResNet18)"
echo "    h^ℓ=layer3(256), h^L=FC(layer4(512)→64)"
$S --dataset flowers102 --case i --mode baseline       --rank 0 --seed $SEED &
$S --dataset flowers102 --case i --mode concat         --rank 1 --seed $SEED &
$S --dataset food101    --case i --mode baseline       --rank 2 --seed $SEED &
$S --dataset food101    --case i --mode concat         --rank 3 --seed $SEED &
wait

$S --dataset flowers102 --case i --mode random_concat  --rank 0 --seed $SEED &
$S --dataset flowers102 --case i --mode inter_only     --rank 1 --seed $SEED &
$S --dataset food101    --case i --mode random_concat  --rank 2 --seed $SEED &
$S --dataset food101    --case i --mode inter_only     --rank 3 --seed $SEED &
wait
echo ">>> Phase 4 DONE $(date)"

# ═══════════════════════════════════════════
# Phase 5: Large (CUB, TinyImageNet) — ResNet50
# ═══════════════════════════════════════════
echo ">>> Phase 5: CUB + TinyImageNet (ResNet50)"
echo "    h^ℓ=layer3(1024), h^L=FC(layer4(2048)→128)"
$S --dataset cub200       --case i --mode baseline       --rank 0 --seed $SEED &
$S --dataset cub200       --case i --mode concat         --rank 1 --seed $SEED &
$S --dataset tinyimagenet --case i --mode baseline       --rank 2 --seed $SEED &
$S --dataset tinyimagenet --case i --mode concat         --rank 3 --seed $SEED &
wait

$S --dataset cub200       --case i --mode random_concat  --rank 0 --seed $SEED &
$S --dataset cub200       --case i --mode inter_only     --rank 1 --seed $SEED &
$S --dataset tinyimagenet --case i --mode random_concat  --rank 2 --seed $SEED &
$S --dataset tinyimagenet --case i --mode inter_only     --rank 3 --seed $SEED &
wait
echo ">>> Phase 5 DONE $(date)"

# ═══════════════════════════════════════════
# Phase 6: Places365 + ImageNet
# ═══════════════════════════════════════════
echo ">>> Phase 6: Places365 + ImageNet"
$S --dataset places365  --case i --mode baseline --rank 0 --seed $SEED &
$S --dataset places365  --case i --mode concat   --rank 1 --seed $SEED &
$S --dataset imagenet1k --case i --mode baseline --rank 2 --seed $SEED &
$S --dataset imagenet1k --case i --mode concat   --rank 3 --seed $SEED &
wait

$S --dataset places365  --case i --mode inter_only     --rank 0 --seed $SEED &
$S --dataset places365  --case i --mode random_concat  --rank 1 --seed $SEED &
$S --dataset imagenet1k --case i --mode inter_only     --rank 2 --seed $SEED &
$S --dataset imagenet1k --case i --mode random_concat  --rank 3 --seed $SEED &
wait
echo ">>> Phase 6 DONE $(date)"

# ═══════════════════════════════════════════
# Phase 7: iNaturalist
# ═══════════════════════════════════════════
echo ">>> Phase 7: iNaturalist"
$S --dataset inat2018 --case iii --mode baseline   --rank 0 --seed $SEED &
$S --dataset inat2018 --case iii --mode concat     --rank 1 --seed $SEED &
$S --dataset inat2021 --case iii --mode baseline   --rank 2 --seed $SEED &
$S --dataset inat2021 --case iii --mode concat     --rank 3 --seed $SEED &
wait

$S --dataset inat2018 --case iii --mode inter_only     --rank 0 --seed $SEED &
$S --dataset inat2018 --case iii --mode random_concat  --rank 1 --seed $SEED &
$S --dataset inat2021 --case iii --mode inter_only     --rank 2 --seed $SEED &
$S --dataset inat2021 --case iii --mode random_concat  --rank 3 --seed $SEED &
wait
echo ">>> Phase 7 DONE $(date)"

# ═══════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════
echo ""
echo "========================================"
echo " RESULTS SUMMARY (Structure B)"
echo "========================================"

for f in output/hnc_v4/hnc.*/results.json; do
    [ -f "$f" ] || continue
    python3 -c "
import json
with open('$f') as fp: r=json.load(fp)
m=r['mode']; d=r['dataset']; c=r['case']
if m=='baseline': s=r['h_L']; fn='h_L'
elif m=='concat': s=r['concat']; fn='cat'
elif m=='random_concat': s=r['h_L']; fn='h_L'
elif m=='inter_only': s=r['h_l']; fn='h_l'
print(f\"{d:<12} case-{c:<3} {m:<12} Acc={r['best_acc']:>6.2f}%  {fn}  D={s['D']:>5}  NC1={s['nc1']:<8}  NC3={s['nc3']:<8}  NCC={s['ncc_acc']}%  params={r['params']:,}\")
" 2>/dev/null
done

echo ""
echo "========================================"
echo " COMPLETE $(date)"
echo "========================================"