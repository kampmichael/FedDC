#! /bin/bash --



lr="0.1"
numclients=150
batch_size=8
numdat=8
numrounds=10000
schedrounds=20000
cuda_device=2

for seed in 1 2 3
do
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_pneum_pytorch.py --run-ablation vanilla_training --optimizer SGD --train-batch-size $batch_size --lr $lr --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --seed $seed | tee Central_Pneum_res_${numclients}cl_n${numdat}_b${batch_size}_lr${lr}_schedule${schedrounds}_r${numrounds}_s${seed}.log

done
