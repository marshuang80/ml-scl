python3 -W ignore main_contrastive.py --wandb_project_name debug \
                                 --log_dir /data/log/contrastive \
				 --match_type any \
                                 --num_epoch 3 \
                                 --model_name densenet121 \
                                 --optimizer adam \
                                 --lr 0.001 \
                                 --batch_size 128 \
                                 --num_workers 8 \
                                 --iters_per_eval 100 \
                                 --resize_shape 150 \
                                 --crop_shape 128 \
                                 --rotation_range 20 \
                                 --img_type Frontal \
                                 --lr_decay 0.1 \
                                 --weight_decay 0.0 \
                                 --momentum 0.9 \
                                 --sgd_dampening 0.0 \
                                 --uncertain ignore \
                                 --temp 0.07 \
                                 --head mlp \
                                 --output_dim 128 \
				 #--use_apex True \
                                 #--gpu_ids 0,1,2,3 \

