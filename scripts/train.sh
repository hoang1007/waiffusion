export WANDB_API_KEY=5a0a277301f0fb20cae141fe944f88db32db5a37
export CUDA_VISIBLE_DEVICES=3

python src/train.py \
    model=ddpm_cifar \
    datamodule=cifar10 \
    trainer=gpu \
    +trainer.precision=16 \
    trainer.max_epochs=20 \
    logger=wandb \
    logger.wandb.group=ddpm_conditional \
    callbacks.image_logger.frequency=50
