CUDA_VISIBLE_DEVICES=0,1,2 nohup torchrun --standalone --nproc_per_node=3 examples/stf_run.py \
    -c config/stf.yaml >stf_e2_lmbda5.0.log