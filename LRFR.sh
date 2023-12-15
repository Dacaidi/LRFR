#!/usr/bin/env bash
GPUID=0
REPEAT=1
############################################
OUTDIR=outputs/cifar100-10
mkdir -p $OUTDIR
python -u main.py -p --prun_ratio 50 -sr --s 0.0001 --schedule 30 60 80 --schedule_pruned 30 60 80 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT --model_optimizer Adam --force_out_dim 0  --first_split_size 10 --other_split_size 10  --batch_size 32 --model_name resnet18 --model_type preact_resnet | tee ${OUTDIR}/output.log

############################################
OUTDIR=outputs/tiny
mkdir -p $OUTDIR
python -u main.py --dataroot ../tiny-imagenet-200/ -p --prun_ratio 50 -sr --s 0.0001 --schedule 30 60 80 --schedule_pruned 30 60 80 --reg_coef 100 --model_lr 5e-5 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4 --gamma 0.5  --svd_thres 10 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset TinyImageNet --gpuid $GPUID --repeat $REPEAT --model_optimizer Adam --force_out_dim 0  --first_split_size 8 --other_split_size 8  --batch_size 16 --model_name resnet18 --model_type preact_resnet | tee ${OUTDIR}/output.log

############################################
OUTDIR=outputs/cifar100-20
mkdir -p $OUTDIR
python -u main.py -p --prun_ratio 50 -sr --s 0.0001 --schedule 30 60 80 --schedule_pruned 30 60 80 --reg_coef 100 --model_lr 1e-4 --head_lr 1e-3 --svd_lr 5e-5 --bn_lr 5e-4  --svd_thres 30 --model_weight_decay 5e-5 --agent_type svd_based --agent_name svd_based --dataset CIFAR100 --gpuid $GPUID --repeat $REPEAT --model_optimizer Adam --force_out_dim 0  --first_split_size 5 --other_split_size 5  --batch_size 16 --model_name resnet18 --model_type preact_resnet | tee ${OUTDIR}/output.log