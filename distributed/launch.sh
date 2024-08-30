#!/bin/bash

# List files before running the training script
echo "Files before training:"
ls -R /demo/Meta-Llama-3.1-8B-Instruct-finetuned

master_addr=$MASTER_ADDR
master_port=$MASTER_PORT
job_n=$WORLD_SIZE
job_id=$RANK

# Echo these if needed
echo ${job_n}
echo ${job_id}
echo ${master_addr}
echo ${master_port}

torchrun --nproc_per_node=1 --nnodes=${job_n} --rdzv_endpoint=${master_addr}:${master_port} --rdzv_backend=c10d distributed.py

# List files after running the training script
echo "Files after training:"
ls -R /demo/Meta-Llama-3.1-8B-Instruct-finetuned
