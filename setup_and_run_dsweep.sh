#!/bin/bash
# setup_and_run_dsweep.sh
# 1. network.py에 bottleneck 추가
# 2. default.py에 bottleneck_dim 추가
# 3. D=8/16/32/64 실험 실행
set -e

cd ~/hierarchical-nc

echo "=== Step 1: Patch network.py ==="

# network.py: __init__에 bottleneck 추가 (reshape 다음, classifier 전)
# 먼저 bottleneck_dim 변수를 다른 config들과 함께 읽도록
python -c "
import re

with open('src/builder/network.py', 'r') as f:
    code = f.read()

# 1) __init__에서 reshape 직후, hier_type 분기 전에 bottleneck 추가
# 찾을 패턴: self.reshape = ... 다음 줄
old = '''        self.reshape = getattr(modules, cfg.reshape.type)(
            cfg, num_features=self.num_features)

        if self.hier_type == 'flat':'''

new = '''        self.reshape = getattr(modules, cfg.reshape.type)(
            cfg, num_features=self.num_features)

        # Bottleneck for D-sweep
        self.bottleneck_dim = getattr(cfg.loss, 'bottleneck_dim', 0)
        if self.bottleneck_dim > 0:
            self._orig_features = self.num_features
            self.bottleneck = nn.Linear(self.num_features, self.bottleneck_dim)
            self.num_features = self.bottleneck_dim

        if self.hier_type == 'flat':'''

if old in code:
    code = code.replace(old, new)
    print('__init__ patched')
else:
    print('WARNING: __init__ pattern not found, trying alternative...')
    # Alternative: insert before flat check
    alt_old = \"\"\"        if self.hier_type == 'flat':
            self.classifier = self._get_classifier(self.num_classes)\"\"\"
    alt_new = \"\"\"        # Bottleneck for D-sweep
        self.bottleneck_dim = getattr(cfg.loss, 'bottleneck_dim', 0)
        if self.bottleneck_dim > 0:
            self._orig_features = self.num_features
            self.bottleneck = nn.Linear(self.num_features, self.bottleneck_dim)
            self.num_features = self.bottleneck_dim

        if self.hier_type == 'flat':
            self.classifier = self._get_classifier(self.num_classes)\"\"\"
    if alt_old in code:
        code = code.replace(alt_old, alt_new)
        print('__init__ patched (alternative)')
    else:
        print('ERROR: Cannot patch __init__')

# 2) extract_feature()에 bottleneck 적용
old_ef = '''    def extract_feature(self, input):
        x = self.backbone(input)
        x = self.pooling(x)
        x = self.reshape(x)
        return x'''

new_ef = '''    def extract_feature(self, input):
        x = self.backbone(input)
        x = self.pooling(x)
        x = self.reshape(x)
        if self.bottleneck_dim > 0:
            x = self.bottleneck(x)
        return x'''

if old_ef in code:
    code = code.replace(old_ef, new_ef)
    print('extract_feature patched')
else:
    print('WARNING: extract_feature pattern not found')

with open('src/builder/network.py', 'w') as f:
    f.write(code)

print('network.py done')
"

echo "=== Step 2: Patch default.py ==="

# default.py에 bottleneck_dim 추가
python -c "
with open('src/config/default.py', 'r') as f:
    code = f.read()

if 'bottleneck_dim' not in code:
    code = code.replace(
        '_C.loss.joint_100way = False',
        '_C.loss.joint_100way = False\n_C.loss.bottleneck_dim = 0'
    )
    print('bottleneck_dim added')
else:
    print('bottleneck_dim already exists')

with open('src/config/default.py', 'w') as f:
    f.write(code)
"

echo "=== Step 3: Clear cache ==="
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "=== Step 4: Quick test ==="
python main/train.py --cfg configs/cifar100/seq_base.yaml \
    ddp False dp False mixed_precision False seed_num 0 rank 0 \
    name "dsweep.test" dataset.hier_type flat \
    loss.bottleneck_dim 16 train.num_epochs 2 \
    proctitle test 2>&1 | tail -3
rm -rf output/cifar100/dsweep.test

echo "=== Step 5: Run D-sweep ==="

O="ddp False dp False mixed_precision False seed_num 0"
C="configs/cifar100/seq_base.yaml"

echo "============================================"
echo " D-Sweep: Flat vs Cos32 vs Direct"
echo " D=8 (K/D=12.5), D=16 (6.25), D=32 (3.13), D=64 (1.56)"
echo " 12 experiments, 3 rounds × 4 GPU, ~2 hours"
echo "============================================"

# R1: D=8 Flat/Cos32/Direct + D=16 Flat
echo ">>> R1: D=8 (K/D=12.5)"
python main/train.py --cfg ${C} ${O} rank 0 \
    name "dsweep.Flat.D8" dataset.hier_type flat \
    loss.bottleneck_dim 8 proctitle flatd8 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "dsweep.Cos32.D8" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 \
    loss.bottleneck_dim 8 proctitle hierd8 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "dsweep.Direct.D8" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 \
    loss.bottleneck_dim 8 proctitle dird8 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "dsweep.Flat.D16" dataset.hier_type flat \
    loss.bottleneck_dim 16 proctitle flatd16 &

wait; echo ">>> R1 DONE"

# R2: D=16 Cos32/Direct + D=32 Flat/Cos32
echo ">>> R2: D=16 + D=32"
python main/train.py --cfg ${C} ${O} rank 0 \
    name "dsweep.Cos32.D16" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 \
    loss.bottleneck_dim 16 proctitle hierd16 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "dsweep.Direct.D16" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 \
    loss.bottleneck_dim 16 proctitle dird16 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "dsweep.Flat.D32" dataset.hier_type flat \
    loss.bottleneck_dim 32 proctitle flatd32 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "dsweep.Cos32.D32" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 \
    loss.bottleneck_dim 32 proctitle hierd32 &

wait; echo ">>> R2 DONE"

# R3: D=32 Direct + D=64 (재확인)
echo ">>> R3: D=32 Direct + D=64"
python main/train.py --cfg ${C} ${O} rank 0 \
    name "dsweep.Direct.D32" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 \
    loss.bottleneck_dim 32 proctitle dird32 &

python main/train.py --cfg ${C} ${O} rank 1 \
    name "dsweep.Flat.D64" dataset.hier_type flat \
    loss.bottleneck_dim 0 proctitle flatd64 &

python main/train.py --cfg ${C} ${O} rank 2 \
    name "dsweep.Cos32.D64" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 loss.cosine_classifier True loss.cosine_scale 32.0 \
    loss.bottleneck_dim 0 proctitle hierd64 &

python main/train.py --cfg ${C} ${O} rank 3 \
    name "dsweep.Direct.D64" dataset.hier_type true_seq_direct \
    loss.lambda_coarse 1.0 \
    loss.bottleneck_dim 0 proctitle dird64 &

wait; echo ">>> R3 DONE"

# RESULTS
echo ""
echo "============================================"
echo " D-SWEEP RESULTS"
echo "============================================"
printf "%-8s | %-8s | %-10s | %-10s | %-10s\n" "D" "K/D" "Flat" "Direct" "Cos32"
printf "%-8s-+-%-8s-+-%-10s-+-%-10s-+-%-10s\n" "--------" "--------" "----------" "----------" "----------"

for dim in 8 16 32 64; do
    kd=$(echo "scale=1; 100/$dim" | bc)
    flat="?"
    direct="?"
    cos32="?"
    
    for d in output/cifar100/dsweep.*.D${dim}; do
        [ -d "$d" ] || continue
        n=$(basename "$d")
        l=$(find "$d" -name "*.log" 2>/dev/null | head -1)
        [ -z "$l" ] && continue
        acc=$(grep "Best Acc" "$l" 2>/dev/null | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
        [ -z "$acc" ] && continue
        
        if [[ "$n" == *"Flat"* ]]; then flat="${acc}%"; fi
        if [[ "$n" == *"Direct"* ]]; then direct="${acc}%"; fi
        if [[ "$n" == *"Cos32"* ]]; then cos32="${acc}%"; fi
    done
    
    printf "%-8s | %-8s | %-10s | %-10s | %-10s\n" "D=$dim" "$kd" "$flat" "$direct" "$cos32"
done

echo "============================================"
echo "Expected: D↓ → K/D↑ → Flat↓↓, Hier↓(약간) → gap 증가"
echo "============================================"