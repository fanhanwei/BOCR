#experiment script for mobilenet v2
if [ $1 = "non" ]; then  
    echo "prune model by BO
    "
    python bo_static.py \
        --job=train \
        --model=mobilenetv2 \
        --dataset=imagenet \
        --preserve_ratio=0.6 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_v2.pth.tar \
        --acc_metric=acc1 \
        --seed=8 \
        --gpu_idx="0" \

elif [ $1 = "static" ]; then  
    echo "prune model by BO with cluster
    "
    python bo_static.py \
        --job=train \
        --model=mobilenetv2 \
        --dataset=imagenet \
        --preserve_ratio=0.6 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_v2.pth.tar \
        --acc_metric=acc1 \
        --seed=10 \
        --static_cluster \
        --n_clusters=6 \
        --simlarity="EU" \
        --gpu_idx="1" \

elif [ $1 = "db" ]; then  
    echo "prune model by BO rollback_direct
    "
    python bo_back.py \
        --job=train \
        --model=mobilenetv2 \
        --dataset=imagenet \
        --preserve_ratio=0.6 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_v2.pth.tar \
        --acc_metric=acc1 \
        --seed=7 \
        --static_cluster \
        --n_clusters=6 \
        --simlarity="EU" \
        --gpu_idx="0" \
    
elif [ $1 = "gb" ]; then  
    echo "prune model by BO gradual rollback
    "
    python bo_back_gradual.py \
        --job=train \
        --model=mobilenetv2 \
        --dataset=imagenet \
        --preserve_ratio=0.6 \
        --lbound=0.1 \
        --rbound=1 \
        --reward=acc_reward \
        --data_root=./dataset/imagenet \
        --ckpt_path=./checkpoints/mobilenet_v2.pth.tar \
        --acc_metric=acc1 \
        --seed=8 \
        --static_cluster \
        --n_clusters=6 \
        --simlarity="EU" \
        --bridge_stage=15 \
        --gpu_idx="1" \
        
else
    echo "
    instruction not recogized!
    "
fi