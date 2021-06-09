{
    set -e
    python -m vit_jax.train \
    --model ViT-B_16 \
    --name debug \
    --logdir logdir_debug \
    --vit_pretrained_dir pretrained_vit_models \
    --dataset imagenet2012 \
    --batch 320
}