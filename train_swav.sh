python3 train_swav.py \
    --data_dir /data/selfsupervision/CheXpert/CheXpert-v1.0/ \
    --image_type Frontal \
    --uncertain ignore \
    --image_size 256 \
    --crop_size 224 \
    --num_workers 8 \
    --batch_size 16 \
    --precision 32 \
    --max_epoch 800 \
    --weights_save_path /data/ckpt \
    --auto_lr_find lr \
    --tpu_cores=8 \
    --optimizer sgd \
    --learning_rate 4.8 \
    --final_lr 0.0048 \
    --start_lr 0.3 \
    --nmb_prototypes 3000 \
    --gpus 0 \
    --dataset chexpert
   
    #--distributed_backend dp \
    #--gpus 4 \

