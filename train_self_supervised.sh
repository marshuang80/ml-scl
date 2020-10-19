python3 train_self_supervised.py \
    --data_dir /data4/selfsupervision/chexpert/CheXpert/CheXpert-v1.0/ \
    --image_type Frontal \
    --uncertain ignore \
    --image_size 256 \
    --crop_size 224 \
    --num_workers 8 \
    --batch_size 16 \
    --deterministic True \
    --benchmark True \
    --precision 16 \
    --distributed_backend dp \
    --gpus 4 \
    --min_epochs 0 \
    --max_epoch 3 \
    --weights_save_path /data4/selfsupervision/log/selfsupervision/ \
    --auto_lr_find lr \

