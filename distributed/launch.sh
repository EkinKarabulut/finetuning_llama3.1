#!/bin/bash

# List files before running the training script
echo "Files before training:"
ls -R /demo/Meta-Llama-3.1-8B-Instruct-finetuned

echo "Using configuration:"
echo "  MASTER_ADDR:   $MASTER_ADDR"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "  WORLD_SIZE:    $WORLD_SIZE"
echo "  RANK:          $RANK"
echo "  PET_NPROC_PER_NODE: $PET_NPROC_PER_NODE"

torchrun --nproc_per_node=1 \
  --nnodes="$WORLD_SIZE" \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_backend=c10d \
  distributed.py

# List files after running the training script
echo "Files after training:"
ls -R /demo/Meta-Llama-3.1-8B-Instruct-finetuned
