#!/bin/bash
O="ddp False dp False mixed_precision False seed_num 0"
G="groupings/seed0"
C="configs/cifar100/v3_unsup_base.yaml"
KM="dataset.grouping_file ${G}/kmeans.json"
CF="dataset.grouping_file ${G}/confusion.json"

r() { echo "=== Round $1 ==="; }

r 1
python main/train.py --cfg $C $O rank 0 name v3.Km-L01.R32 $KM loss.lambda_coarse 0.1 proctitle r1a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L005.R32 $KM loss.lambda_coarse 0.05 proctitle r1b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L01.R32 $CF loss.lambda_coarse 0.1 proctitle r1c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L005.R32 $CF loss.lambda_coarse 0.05 proctitle r1d &
wait
r 2
python main/train.py --cfg $C $O rank 0 name v3.Km-L02.R32 $KM loss.lambda_coarse 0.2 proctitle r2a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L001.R32 $KM loss.lambda_coarse 0.01 proctitle r2b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L02.R32 $CF loss.lambda_coarse 0.2 proctitle r2c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L001.R32 $CF loss.lambda_coarse 0.01 proctitle r2d &
wait
r 3
python main/train.py --cfg $C $O rank 0 name v3.Km-L03D.R32 $KM loss.lambda_coarse 0.3 loss.lambda_decay True proctitle r3a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L01D.R32 $KM loss.lambda_coarse 0.1 loss.lambda_decay True proctitle r3b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L03D.R32 $CF loss.lambda_coarse 0.3 loss.lambda_decay True proctitle r3c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L01D.R32 $CF loss.lambda_coarse 0.1 loss.lambda_decay True proctitle r3d &
wait
r 4
python main/train.py --cfg $C $O rank 0 name v3.Km-L005D.R32 $KM loss.lambda_coarse 0.05 loss.lambda_decay True proctitle r4a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L02D.R32 $KM loss.lambda_coarse 0.2 loss.lambda_decay True proctitle r4b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L005D.R32 $CF loss.lambda_coarse 0.05 loss.lambda_decay True proctitle r4c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L02D.R32 $CF loss.lambda_coarse 0.2 loss.lambda_decay True proctitle r4d &
wait
r 5
python main/train.py --cfg $C $O rank 0 name v3.Km-L001D.R32 $KM loss.lambda_coarse 0.01 loss.lambda_decay True proctitle r5a &
python main/train.py --cfg $C $O rank 1 name v3.Conf-L001D.R32 $CF loss.lambda_coarse 0.01 loss.lambda_decay True proctitle r5b &
python main/train.py --cfg $C $O rank 2 name v3.Km-L03Det.R32 $KM loss.lambda_coarse 0.3 loss.coarse_detach True proctitle r5c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L03Det.R32 $CF loss.lambda_coarse 0.3 loss.coarse_detach True proctitle r5d &
wait
r 6
python main/train.py --cfg $C $O rank 0 name v3.Km-L01Det.R32 $KM loss.lambda_coarse 0.1 loss.coarse_detach True proctitle r6a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L005Det.R32 $KM loss.lambda_coarse 0.05 loss.coarse_detach True proctitle r6b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L01Det.R32 $CF loss.lambda_coarse 0.1 loss.coarse_detach True proctitle r6c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L005Det.R32 $CF loss.lambda_coarse 0.05 loss.coarse_detach True proctitle r6d &
wait
r 7
python main/train.py --cfg $C $O rank 0 name v3.Km-L02Det.R32 $KM loss.lambda_coarse 0.2 loss.coarse_detach True proctitle r7a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L001Det.R32 $KM loss.lambda_coarse 0.01 loss.coarse_detach True proctitle r7b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L02Det.R32 $CF loss.lambda_coarse 0.2 loss.coarse_detach True proctitle r7c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L001Det.R32 $CF loss.lambda_coarse 0.01 loss.coarse_detach True proctitle r7d &
wait
r 8
python main/train.py --cfg $C $O rank 0 name v3.Km-L03DD.R32 $KM loss.lambda_coarse 0.3 loss.lambda_decay True loss.coarse_detach True proctitle r8a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L01DD.R32 $KM loss.lambda_coarse 0.1 loss.lambda_decay True loss.coarse_detach True proctitle r8b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L03DD.R32 $CF loss.lambda_coarse 0.3 loss.lambda_decay True loss.coarse_detach True proctitle r8c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L01DD.R32 $CF loss.lambda_coarse 0.1 loss.lambda_decay True loss.coarse_detach True proctitle r8d &
wait
r 9
python main/train.py --cfg $C $O rank 0 name v3.Km-L005DD.R32 $KM loss.lambda_coarse 0.05 loss.lambda_decay True loss.coarse_detach True proctitle r9a &
python main/train.py --cfg $C $O rank 1 name v3.Km-L02DD.R32 $KM loss.lambda_coarse 0.2 loss.lambda_decay True loss.coarse_detach True proctitle r9b &
python main/train.py --cfg $C $O rank 2 name v3.Conf-L005DD.R32 $CF loss.lambda_coarse 0.05 loss.lambda_decay True loss.coarse_detach True proctitle r9c &
python main/train.py --cfg $C $O rank 3 name v3.Conf-L02DD.R32 $CF loss.lambda_coarse 0.2 loss.lambda_decay True loss.coarse_detach True proctitle r9d &
wait
r 10
python main/train.py --cfg $C $O rank 0 name v3.Km-L001DD.R32 $KM loss.lambda_coarse 0.01 loss.lambda_decay True loss.coarse_detach True proctitle r10a &
python main/train.py --cfg $C $O rank 1 name v3.Conf-L001DD.R32 $CF loss.lambda_coarse 0.01 loss.lambda_decay True loss.coarse_detach True proctitle r10b &
wait

echo ""
echo "========== FULL RESULTS =========="
printf "%-30s %8s\n" "Flat" "71.52%"
for d in output/cifar100/v3.Km-* output/cifar100/v3.Conf-* output/cifar100/v3.Unsup*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    l=$(find "$d" -name "*.log" 2>/dev/null | head -1)
    [ -z "$l" ] && continue
    a=$(grep "Best Acc" "$l" | tail -1 | grep -oP '[0-9]+\.[0-9]+' | tail -1)
    printf "%-30s %8s\n" "$n" "${a}%"
done
echo "=================================="
