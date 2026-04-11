#!/bin/bash
# ============================================================
# CIFAR-100 추가 실험 (서버에서 실행)
# cd ~/hierarchical-nc && bash run_remaining_cifar.sh
#
# Force 구성:
#   intra tightening = center_loss_weight (신규, features→class mean)
#   inter separation = nc_reg_weight (기존, class mean→ETF ideal)
#
# 실험 목록:
#   Round 1: 2-Stage S1 + Supervised + Km-L02 seed1,2
#   Round 2: 2-Stage S2 (S1 weights 로드)
#   Round 3: Center loss sweep (intra)
#   Round 4: NC reg sweep (inter)
#   Round 5: Center × NC reg 2D grid
#   Round 6: Best combo + decay/detach/confusion
#
# 예상: ~4시간 (밤새 돌리기)
# ============================================================
set -e

O="ddp False dp False mixed_precision False"
G="groupings/seed0"
C="configs/cifar100/v3_unsup_base.yaml"

# ============================================================
# STEP 0: Trainer 패치
# ============================================================
echo "[Step 0] Patching trainer..."

python << 'PYFIX'
with open('src/core/trainer.py', 'r') as f:
    code = f.read()

changed = False

# --- 1. two_stage_mode ---
if 'two_stage_mode' not in code:
    code = code.replace(
        "self.coarse_detach = getattr(cfg.loss, 'coarse_detach', False)",
        """self.coarse_detach = getattr(cfg.loss, 'coarse_detach', False)
        self.two_stage_mode = getattr(cfg.loss, 'two_stage_mode', 'joint')""")
    code = code.replace(
        "if self.lambda_decay:",
        """if self.two_stage_mode == 'coarse_only':
            return 1.0
        if self.two_stage_mode == 'fine_only':
            return 0.0
        if self.lambda_decay:""")
    changed = True
    print("  Added two_stage_mode")

# --- 2. center_loss (intra tightening) ---
if 'center_loss_weight' not in code:
    code = code.replace(
        "self.nc_reg_weight = getattr(cfg.loss, 'nc_reg_weight', 0.0)",
        """self.nc_reg_weight = getattr(cfg.loss, 'nc_reg_weight', 0.0)
        self.center_loss_weight = getattr(cfg.loss, 'center_loss_weight', 0.0)""")

    # NC regularizer 뒤, pred 앞에 center loss 삽입
    code = code.replace(
        "        pred = torch.argmax(fine_logits, 1)\n        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]\n        return loss, acc\n    def mixup",
        """            # Center loss (intra-class tightening)
            if self.center_loss_weight > 0:
                c_loss = self._center_loss(features, targets)
                loss = loss + self.center_loss_weight * c_loss
        pred = torch.argmax(fine_logits, 1)
        acc = accuracy(pred.cpu().numpy(), targets.cpu().numpy())[0]
        return loss, acc

    def _center_loss(self, features, targets):
        \"\"\"Pull features toward EMA class centers (intra-class tightening).\"\"\"
        K = self.num_classes
        D = features.shape[1]
        if self._class_means is None:
            self._class_means = torch.zeros(K, D, device=features.device)
            self._class_counts = torch.zeros(K, device=features.device)
        with torch.no_grad():
            for c in range(K):
                mask = (targets == c)
                if mask.sum() > 0:
                    batch_mean = features[mask].mean(0)
                    self._class_means[c] = 0.9 * self._class_means[c] + 0.1 * batch_mean
                    self._class_counts[c] += mask.sum()
        centers = self._class_means[targets].detach()
        loss = ((features - centers) ** 2).sum(1).mean()
        return loss

    def mixup""")
    changed = True
    print("  Added center_loss")

if changed:
    with open('src/core/trainer.py', 'w') as f:
        f.write(code)
    print("  trainer.py saved!")
else:
    print("  Already patched")

# default.py
with open('src/config/default.py', 'r') as f:
    d = f.read()
added = False
for key, line in [
    ('two_stage_mode', '_C.loss.two_stage_mode = "joint"'),
    ('center_loss_weight', '_C.loss.center_loss_weight = 0.0'),
]:
    if key not in d:
        d = d.replace('_C.loss.coarse_detach = False',
                      f'_C.loss.coarse_detach = False\n{line}')
        added = True
        print(f"  Added {key} to default.py")
if added:
    with open('src/config/default.py', 'w') as f:
        f.write(d)
print("Done!")
PYFIX

find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

echo "=== Verify ==="
grep "two_stage_mode\|center_loss_weight" src/core/trainer.py | head -4
grep "two_stage_mode\|center_loss_weight" src/config/default.py
echo ""

# ============================================================
# ROUND 1: 2-Stage S1 + Supervised + Seeds (4 GPU, ~1h)
# ============================================================
echo "========== ROUND 1: Stage1 + Sup + Seeds =========="

python main/train.py --cfg ${C} ${O} seed_num 0 rank 0 \
    name "v3.Km-2Stage-S1.R32" \
    dataset.grouping_file "${G}/kmeans.json" \
    loss.two_stage_mode "coarse_only" \
    train.num_epochs 100 \
    proctitle s1 &

python main/train.py --cfg ${C} ${O} seed_num 0 rank 1 \
    name "v3.Supervised.R32" \
    dataset.grouping_file "none" \
    proctitle sup &

python main/train.py --cfg ${C} ${O} seed_num 1 rank 2 \
    name "v3.Km-L02-s1.R32" \
    dataset.grouping_file "${G}/kmeans.json" \
    loss.lambda_coarse 0.2 \
    proctitle ks1 &

python main/train.py --cfg ${C} ${O} seed_num 2 rank 3 \
    name "v3.Km-L02-s2.R32" \
    dataset.grouping_file "${G}/kmeans.json" \
    loss.lambda_coarse 0.2 \
    proctitle ks2 &

wait
echo "[Round 1 DONE]"

# ============================================================
# ROUND 2: Stage2 — S1 weights 로드, fine only (1 GPU, ~40m)
# ============================================================
echo "========== ROUND 2: Stage2 (fine only) =========="

S1_CKPT="output/cifar100/v3.Km-2Stage-S1.R32/seed000/models/best_model.pth"
if [ ! -f "${S1_CKPT}" ]; then
    S1_CKPT=$(ls output/cifar100/v3.Km-2Stage-S1.R32/seed000/models/epoch_*.pth 2>/dev/null | sort -V | tail -1)
fi

if [ -f "${S1_CKPT}" ]; then
    python main/train.py --cfg ${C} ${O} seed_num 0 rank 0 \
        name "v3.Km-2Stage-S2.R32" \
        dataset.grouping_file "${G}/kmeans.json" \
        loss.two_stage_mode "fine_only" \
        pretrained "${S1_CKPT}" \
        proctitle s2
    echo "[Round 2 DONE]"
else
    echo "[ERROR] Stage1 checkpoint not found!"
fi

# ============================================================
# ROUND 3: Center loss sweep — intra tightening (4 GPU, ~40m)
# Base: Km-L02 (best NC setting)
# ============================================================
echo "========== ROUND 3: Center loss (intra) sweep =========="
KM="dataset.grouping_file ${G}/kmeans.json loss.lambda_coarse 0.2"

python main/train.py --cfg $C $O rank 0 name "v3.Km-L02-ctr0001.R32" $KM loss.center_loss_weight 0.0001 proctitle c0001 &
python main/train.py --cfg $C $O rank 1 name "v3.Km-L02-ctr001.R32"  $KM loss.center_loss_weight 0.001  proctitle c001 &
python main/train.py --cfg $C $O rank 2 name "v3.Km-L02-ctr01.R32"   $KM loss.center_loss_weight 0.01   proctitle c01 &
python main/train.py --cfg $C $O rank 3 name "v3.Km-L02-ctr1.R32"    $KM loss.center_loss_weight 0.1    proctitle c1 &
wait
echo "[Round 3 DONE]"

# ============================================================
# ROUND 4: NC reg sweep — inter separation (4 GPU, ~40m)
# ============================================================
echo "========== ROUND 4: NC reg (inter) sweep =========="

python main/train.py --cfg $C $O rank 0 name "v3.Km-L02-ncr0001.R32" $KM loss.nc_reg_weight 0.0001 proctitle n0001 &
python main/train.py --cfg $C $O rank 1 name "v3.Km-L02-ncr001.R32"  $KM loss.nc_reg_weight 0.001  proctitle n001 &
python main/train.py --cfg $C $O rank 2 name "v3.Km-L02-ncr01.R32"   $KM loss.nc_reg_weight 0.01   proctitle n01 &
python main/train.py --cfg $C $O rank 3 name "v3.Km-L02-ncr1.R32"    $KM loss.nc_reg_weight 0.1    proctitle n1 &
wait
echo "[Round 4 DONE]"

# ============================================================
# ROUND 5: Center × NC reg 2D grid (4 GPU, ~40m)
# ============================================================
echo "========== ROUND 5: Center x NC reg grid =========="

python main/train.py --cfg $C $O rank 0 name "v3.Km-L02-c001n001.R32" $KM loss.center_loss_weight 0.001 loss.nc_reg_weight 0.001 proctitle cn1 &
python main/train.py --cfg $C $O rank 1 name "v3.Km-L02-c01n001.R32"  $KM loss.center_loss_weight 0.01  loss.nc_reg_weight 0.001 proctitle cn2 &
python main/train.py --cfg $C $O rank 2 name "v3.Km-L02-c001n01.R32"  $KM loss.center_loss_weight 0.001 loss.nc_reg_weight 0.01  proctitle cn3 &
python main/train.py --cfg $C $O rank 3 name "v3.Km-L02-c01n01.R32"   $KM loss.center_loss_weight 0.01  loss.nc_reg_weight 0.01  proctitle cn4 &
wait
echo "[Round 5 DONE]"

# ============================================================
# ROUND 6: Best combo + decay/detach + Confusion (4 GPU, ~40m)
# ============================================================
echo "========== ROUND 6: Combos + Confusion =========="
CF="dataset.grouping_file ${G}/confusion.json loss.lambda_coarse 0.2"

python main/train.py --cfg $C $O rank 0 name "v3.Km-L02-c01n001D.R32"    $KM loss.center_loss_weight 0.01 loss.nc_reg_weight 0.001 loss.lambda_decay True proctitle d1 &
python main/train.py --cfg $C $O rank 1 name "v3.Km-L02-c01n001Det.R32"  $KM loss.center_loss_weight 0.01 loss.nc_reg_weight 0.001 loss.coarse_detach True proctitle d2 &
python main/train.py --cfg $C $O rank 2 name "v3.Conf-L02-c01n001.R32"   $CF loss.center_loss_weight 0.01 loss.nc_reg_weight 0.001 proctitle d3 &
python main/train.py --cfg $C $O rank 3 name "v3.Conf-L02-ctr01.R32"     $CF loss.center_loss_weight 0.01 proctitle d4 &
wait
echo "[Round 6 DONE]"

# ============================================================
# RESULTS
# ============================================================
echo ""
echo "============================================"
echo " ALL RESULTS"
echo "============================================"
printf "%-45s %8s\n" "Model" "Acc%"
printf "%-45s %8s\n" "-----" "----"
printf "%-45s %8s\n" "Flat (ref)" "71.52%"
printf "%-45s %8s\n" "Km-L02 seed0 (ref)" "71.43%"
printf "%-45s %8s\n" "Km-L005D (ref, acc winner)" "71.59%"

for d in output/cifar100/v3.Km-2Stage* \
         output/cifar100/v3.Supervised* \
         output/cifar100/v3.Km-L02-s*.R32 \
         output/cifar100/v3.Km-L02-ctr* \
         output/cifar100/v3.Km-L02-ncr* \
         output/cifar100/v3.Km-L02-c0* \
         output/cifar100/v3.Km-L02-c01n* \
         output/cifar100/v3.Conf-L02-*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    l=$(find "$d" -name "*.log" 2>/dev/null | head -1)
    [ -z "$l" ] && continue
    a=$(grep "Best Acc" "$l" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "%-45s %8s\n" "$n" "${a:-?}%"
done | sort -t'%' -k2 -rn

echo "============================================"
echo ""
echo "다음: top 5 모델에 NC stats 수집"
