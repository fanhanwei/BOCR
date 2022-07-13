#experiment script for mobilenet v1
if [ $1 = "non" ]; then  
    echo "prune model by naive BO
    "
    python bo_static.py \
        --job=train \
        --model=mobilenet \
        --dataset=imagenet \
        --preserve_ratio=0.5 \
        --lbound=0.2 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
        --acc_metric=acc1 \
        --seed=1 \

elif [ $1 = "static" ]; then  
    echo "prune model by BO with cluster
    "
    python bo_static.py \
        --job=train \
        --model=mobilenet \
        --dataset=imagenet \
        --preserve_ratio=0.5 \
        --lbound=0.2 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
        --acc_metric=acc1 \
        --seed=2 \
        --static_cluster \
        --n_clusters=6 \
        --simlarity="EU"\
        --gpu_idx="0" \

elif [ $1 = "db" ]; then  
    echo "prune model by BO direct rollback
    "
    python bo_back.py \
        --job=train \
        --model=mobilenet \
        --dataset=imagenet \
        --preserve_ratio=0.5 \
        --lbound=0.2 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
        --acc_metric=acc1 \
        --seed=161 \
        --static_cluster \
        --n_clusters=6 \
        --simlarity="EU"\
        --gpu_idx="0" \

elif [ $1 = "gb" ]; then  
    echo "prune model by BO gradual rollback
    "
    python bo_back_gradual.py \
        --job=train \
        --model=mobilenet \
        --dataset=imagenet \
        --preserve_ratio=0.5 \
        --lbound=0.2 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_imagenet.pth.tar \
        --acc_metric=acc1 \
        --seed=1 \
        --static_cluster \
        --n_clusters=6 \
        --simlarity="EU"\
        --bridge_stage=9 \
        --gpu_idx="1" \

else
    echo "
    instruction not recogized!
    "
fi