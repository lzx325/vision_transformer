python -m vit_jax.train \
    --model ViT-B_16 \
    --name "$exp_name" \
    --train_dir debug \
    --vit_pretrained_dir pretrained_vit_models \
    --dataset imagenet2012 \
    --batch 1280 \
    --eval_every 500