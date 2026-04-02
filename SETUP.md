# NC Pilot Patch — 적용 가이드

## 변경 요약

선배님 image-classification 코드에 **3가지 classifier 타입**을 추가합니다.

| 타입 | config 값 | 출력 | Metric | 설명 |
|------|----------|------|--------|------|
| Flat | (기존) | 100-way | fine_100 acc | baseline |
| Sequential | `hier_type: 'default'` or `'sequential'` | (20,5) | joint acc | 선배님 original |
| **Seq Residual** | `hier_type: 'sequential_residual'` | (20,5) | joint acc | h+α·sg(Wp) |
| **Factorized** | `hier_type: 'factorized'` | (100,20) | fine_100 + coarse_20 | 건우님 방식 |

기존 config 파일은 **전부 그대로 동작**합니다 (backward compatible).

## 적용 방법

```bash
# 1. 서버에서 선배님 레포로 이동
cd ~/image-classification

# 2. patch 파일들 복사 (덮어쓰기)
cp patch/src/modules/hier_ops.py     src/modules/hier_ops.py
cp patch/src/modules/__init__.py     src/modules/__init__.py
cp patch/src/builder/network.py      src/builder/network.py
cp patch/src/config/default.py       src/config/default.py
cp patch/src/core/trainer.py         src/core/trainer.py
cp patch/src/core/function.py        src/core/function.py

cp patch/configs/cifar100/ce_hiercifar100_vgg11_factorized.yaml      configs/cifar100/
cp patch/configs/cifar100/ce_hiercifar100_vgg11_factorized_rand.yaml configs/cifar100/
cp patch/configs/cifar100/ce_hiercifar100_vgg11_seqres.yaml          configs/cifar100/

cp patch/RunPilotCompare.sh .
chmod +x RunPilotCompare.sh

# 3. 실행 (GPU 0, seed 0)
bash RunPilotCompare.sh 0 0
```

## 수정된 파일 목록

| 파일 | 변경 | 내용 |
|------|------|------|
| `src/modules/hier_ops.py` | **새 파일** | FactorizedPerGroup classifier |
| `src/modules/__init__.py` | 수정 | `from .hier_ops import *` 추가 |
| `src/builder/network.py` | 수정 | hier_type에 따른 classify() 분기 |
| `src/config/default.py` | 수정 | `hier_type`, `lambda_coarse` 추가 |
| `src/core/trainer.py` | 수정 | factorized 출력 처리 |
| `src/core/function.py` | 수정 | fine/coarse acc 동시 보고 |

## 계산 흐름 비교

### Sequential (선배님)
```
h(4096) → W1(4096×20) → logit_20 → softmax → p1(20)
                                                  ↓ stop-grad
                             p1 @ W1 → h2(4096) → W2(4096×5) → logit_5
Loss = CE(logit_20, coarse_label) + CE(logit_5, super_label)
Acc  = (pred_20 correct) AND (pred_5 correct)
```

### Sequential Residual (NEW)
```
h(4096) → W1(4096×20) → logit_20 → softmax → p1(20)
                                                  ↓ stop-grad
                    h2 = h + α·(p1 @ W1)  ← h 보존!
                             h2(4096) → W2(4096×5) → logit_5
Loss = CE(logit_20, coarse_label) + CE(logit_5, super_label)
Acc  = (pred_20 correct) AND (pred_5 correct)
```

### Factorized (NEW)
```
h(4096) → W_c(4096×20) → coarse_logit(20)
       → W_g(4096×20×5) → fine_per_group(20×5)
                  ↓ assemble
       logit_100[c] = coarse[g(c)] + fine_g[f(c)]

Loss = CE(logit_100, fine_label) + λ·CE(coarse_logit, coarse_label)
Acc  = fine_100 accuracy
```

## 예상 결과

| Model | Metric | 예상 범위 | 비고 |
|-------|--------|----------|------|
| Flat | fine_100 | ~67% | 기존 결과와 동일해야 함 |
| Sequential | joint_20+5 | ~78% | 선배님 결과 재현 |
| Seq+Rand | joint_20+5 | ~66% | random 대조군, 낮을 것 |
| SeqResidual | joint_20+5 | ~80%? | **residual이 도움되는지 핵심 실험** |
| Factorized | fine_100 | ~65-67% | flat과 비교 |
| Fact+Rand | fine_100 | ~63-65% | semantic grouping 효과 확인 |
