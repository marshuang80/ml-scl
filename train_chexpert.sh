python3 -W ignore main_chexpert.py --wandb_project_name debug \
                                 --log_dir /data4/selfsupervision/log \
                                 --num_epoch 3 \
                                 --model_name densenet121 \
                                 --optimizer adam \
                                 --lr 0.001 \
                                 --batch_size 16 \
                                 --num_workers 8 \
                                 --iters_per_eval 100 \
                                 --gpu_ids 0,1,2 \
                                 --resize_shape 320 \
                                 --crop_shape 320 \
                                 --rotation_range 20 \
                                 --img_type Frontal \
                                 --lr_decay 0.1 \
                                 --weight_decay 0.0 \
                                 --momentum 0.9 \
                                 --sgd_dampening 0.0 \
                                 --uncertain ignore \
                                 --threshold 0.5
