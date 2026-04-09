#!/bin/bash
# ============================================================
# SETUP: 선배님 코드베이스에 unsupervised hierarchy 실험 추가
# 이 스크립트를 선배님 레포 루트에서 실행하면 전부 세팅됩니다.
# 
# Usage: bash setup_unsup.sh
# ============================================================

set -e

echo "=== Setting up unsupervised hierarchy experiments ==="

# 0. 디렉토리 생성
mkdir -p tools groupings nc_stats

# 1. 새 파일 복사 (이 스크립트와 같은 폴더에 있다고 가정)
PATCH_DIR="$(dirname "$0")"

echo "[1/6] Copying new files..."
cp "${PATCH_DIR}/discover_hierarchy.py"      tools/
cp "${PATCH_DIR}/collect_nc_stats_cifar.py"   tools/
cp "${PATCH_DIR}/unsup_hier_dataset.py"       src/dataset/
cp "${PATCH_DIR}/hier_ops_v2.py"              src/modules/hier_ops.py
cp "${PATCH_DIR}/v3_unsup_base.yaml"          configs/cifar100/

# 2. dataset __init__.py에 import 추가
echo "[2/6] Patching src/dataset/__init__.py..."
if ! grep -q "UnsupHierCIFAR100" src/dataset/__init__.py; then
    echo "from .unsup_hier_dataset import UnsupHierCIFAR100" >> src/dataset/__init__.py
    echo "  Added UnsupHierCIFAR100 import"
else
    echo "  Already patched"
fi

# 3. config default.py에 새 key 추가
echo "[3/6] Patching src/config/default.py..."
if ! grep -q "grouping_file" src/config/default.py; then
    # _C.dataset 섹션에 추가
    sed -i '/^_C.dataset.hier_type/a _C.dataset.grouping_file = "none"' src/config/default.py
    echo "  Added grouping_file"
else
    echo "  grouping_file already exists"
fi

if ! grep -q "lambda_warmup" src/config/default.py; then
    # _C.loss 섹션에 추가
    sed -i '/^_C.loss.soft_beta/a _C.loss.lambda_warmup = False' src/config/default.py
    echo "  Added lambda_warmup"
else
    echo "  lambda_warmup already exists"
fi

# 4. network.py에 JSON grouping 로드 추가
echo "[4/6] Patching src/builder/network.py..."
cp "${PATCH_DIR}/network_patched.py" src/builder/network.py
echo "  Replaced network.py"

# v3 폴더가 있으면 거기도 패치
if [ -d "v3/src/builder" ]; then
    cp "${PATCH_DIR}/network_patched.py" v3/src/builder/network.py
    echo "  Also patched v3/src/builder/network.py"
fi

# 5. trainer에 λ warmup 추가
echo "[5/6] Patching src/core/trainer.py..."
cp src/core/trainer.py src/core/trainer_backup.py
cp "${PATCH_DIR}/trainer_extended.py" src/core/trainer.py
echo "  Replaced trainer.py (backup: trainer_backup.py)"

if [ -d "v3/src/core" ]; then
    cp v3/src/core/trainer.py v3/src/core/trainer_backup.py
    cp "${PATCH_DIR}/trainer_extended.py" v3/src/core/trainer.py
    echo "  Also patched v3/src/core/trainer.py"
fi

# 6. dependency 확인
echo "[6/6] Checking dependencies..."
python -c "import sklearn; print('  scikit-learn OK')" 2>/dev/null || echo "  NEED: pip install scikit-learn"
python -c "import scipy; print('  scipy OK')" 2>/dev/null || echo "  NEED: pip install scipy"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. pip install scikit-learn scipy  (if needed)"
echo "  2. bash RunUnsupervised.sh <GPU_ID> <SEED> resnet32"
echo ""
