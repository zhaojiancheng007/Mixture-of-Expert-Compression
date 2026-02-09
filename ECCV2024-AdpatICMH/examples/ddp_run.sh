CUDA_VISIBLE_DEVICES=0,1,2 nohup torchrun --standalone --nproc_per_node=3 examples/moe_ddprun.py \
    -c config/moe.yaml >cvpr_e2_dec_lmbda5.0.log