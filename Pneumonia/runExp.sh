#! /bin/bash --



lr="0.1"
numclients=150
batch_size=8
numdat=8
numrounds=10000
schedrounds=20000
cuda_device=7

for seed in 1 2 3
do
    daisy=1
    avg=10
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_pneum_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr $lr --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg | tee CompExp_Pneum_res_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr${lr}_schedule${schedrounds}_r${numrounds}_s${seed}.log

    daisy=1
    avg=20001
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_pneum_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr $lr --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg | tee CompExp_Pneum_res_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr${lr}_schedule${schedrounds}_r${numrounds}_s${seed}.log

    daisy=20001
    avg=10
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_pneum_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr $lr --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg | tee CompExp_Pneum_res_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr${lr}_schedule${schedrounds}_r${numrounds}_s${seed}.log

    daisy=20001
    avg=1
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_pneum_pytorch.py --optimizer SGD --train-batch-size $batch_size --lr $lr --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --daisy-rounds $daisy --aggregate-rounds $avg | tee CompExp_Pneum_res_${numclients}cl_n${numdat}_b${batch_size}_d${daisy}_a${avg}_lr${lr}_schedule${schedrounds}_r${numrounds}_s${seed}.log
done
