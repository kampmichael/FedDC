#! /bin/bash --


model="MRInet"

lr="0.1"
numclients=25
batch_size=8
numdat=8
numrounds=10000
schedrounds=5000
cuda_device=2
weight_decay=0.01

for seed in 1 2 3
do
	CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cuda_device} python3 -u feddc_MRI_pytorch.py --run-ablation vanilla_training --optimizer SGD --train-batch-size $batch_size --lr 0.1 --lr-schedule-ep ${schedrounds} --lr-change-rate 0.5 --num-clients $numclients --num-rounds $numrounds --num-samples-per-client $numdat --report-rounds 250 --seed $seed | tee Central_MRI_${numclients}cl_n${numdat}_b${batch_size}_lr0_1_schedule${schedrounds}_r${numrounds}_s${seed}.log

done
