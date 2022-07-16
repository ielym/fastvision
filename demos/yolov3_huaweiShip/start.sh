CUDA_VISIBLE_DEVICES=0,1 nohup python -u run.py --input_size 608 --batch_size 32 --init_lr 1e-3 --start_epoch 0 --warmup_epoch 10 --no_aug_epoch 10 --total_epoch 300 > logs/log.txt 2>&1 &
