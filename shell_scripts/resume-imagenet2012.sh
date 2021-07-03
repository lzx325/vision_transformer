python -m vit_jax.train \
    --mode resume \
    --model ViT-B_16_pretrain \
    --exp_name "$exp_name" \
    --train_dir train_dir \
    --vit_pretrained_dir pretrained_vit_models \
    --dataset imagenet2012 \
    --batch 1280 \
    --base_lr 1e-3 \
    --decay_type cosine \
    --linear_end 1e-5 \
    --warmup_steps 500 \
    --eval_every 1000 \
    --total_steps 80000 \
    --weight_decay 0.00 \
    --resume_dir train_dir/imagenet2012_pretrain_2/2021-06-13-13:00:27 \
    --resume_from step-44000.npz
