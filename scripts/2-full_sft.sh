nohup torchrun \
    --nproc_per_node 8 \
    3-full_sft.py \
    --use_wandb \
    --ddp \
    --epochs 3 \
    --out_dir /home/jovyan/zlcode/minimind/out_dir/sft \
    2>&1 > /home/jovyan/zlcode/minimind/log/full_sft.log &
