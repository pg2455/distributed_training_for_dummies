#!/bin/sh
set -e

if [ "$RUN_MODE" = "comm" ]; then
    python sample_comm_primitive.py

elif [ "$RUN_MODE" = "train" ]; then
    python sample_train.py

elif [ "$RUN_MODE" = "data_parallel_naive" ]; then
    python data_parallel/train_naive.py

elif [ "$RUN_MODE" = "data_parallel_naive_grad_accumulation" ]; then
    python data_parallel/train_naive_grad_accumulation.py --grad_acc_steps 2

elif [ "$RUN_MODE" = "data_parallel_overlap" ]; then
    python data_parallel/train_overlap.py --grad_acc_steps 2

elif [ "$RUN_MODE" = "data_parallel_bucket_overlap" ]; then
    python data_parallel/train_bucket_overlap.py --grad_acc_steps 3

elif [ "$RUN_MODE" = "tensor_parallel" ]; then
    python tensor_parallel/train.py
else
    echo "Please set RUN_MODE to 'comm' or 'train'"
    exit 1
fi
