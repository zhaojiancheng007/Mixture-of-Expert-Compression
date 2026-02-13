#!/bin/bash
# ==============================================================
#  DDP multi-GPU training with torchrun
#  Usage:
#    bash examples/run.sh                      # use all visible GPUs
#    CUDA_VISIBLE_DEVICES=0,1 bash examples/run.sh   # use GPU 0,1
#    CUDA_VISIBLE_DEVICES=0 bash examples/run.sh     # single GPU (DDP disabled automatically)
# ==============================================================

CONFIG="config/moe_st_psnr.yaml"
LOG="TIC_MoE_spatial_lmbda5e3.log"

# Automatically detect number of visible GPUs
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")
echo "Detected ${NGPU} GPU(s)"

if [ "$NGPU" -gt 1 ]; then
    echo "Launching DDP training with torchrun on ${NGPU} GPUs..."
    nohup torchrun \
        --standalone \
        --nproc_per_node=${NGPU} \
        examples/moecodec_psnr.py \
        -c ${CONFIG} \
        > ${LOG} 2>&1 &
else
    echo "Single-GPU mode..."
    nohup python examples/moecodec_psnr.py \
        -c ${CONFIG} \
        > ${LOG} 2>&1 &
fi

echo "PID: $!"
echo "Log: ${LOG}"
