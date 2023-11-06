# NOTE: use display_id = 0 to disable visdom!!!
train_model()
{
    echo "-----------Training TPSeNCE with [dataname:$1, name:$2, load_size:$3, crop_size:$4, lambda_TRI:$5, sem_metrics:$6, out_file:$7]----------->"
    nohup \
    python -u train.py \
        --batch_size 32 \
        --gpu_ids '0,1,2,3,4,5,6,7' \
        --n_epochs 100 \
        --n_epochs_decay 100 \
        --dataroot /root/autodl-tmp/Datasets/$1 \
        --name $2 \
        --preprocess scale_width_and_crop \
        --load_size $3 \
        --crop_size $4 \
        --lambda_TRI $5 \
        --dataset_mode quad \
        --cost_type mixed \
        --display_id 0 \
        --sem_metrics $6 \
        > $7 2>&1 &
}

# training BDD
train_model 'bdd100k_1_20' 'bdd100k_1_20' 640 256 1.0 'mPA' 'bdd100k_rainy.out'

# training INIT
# train_model 'INIT_rainy' 'INIT_rainy_full' 572 256 1.0 'INIT_rainy.out'

# # train boreas
# train_model 'boreas_snowy' 'boreas_snowy' 430 256 1.0 'mPA' 'boreas_snowy.out'