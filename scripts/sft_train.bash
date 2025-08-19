export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=1,2,3,4,6,7 deepspeed trainer/train_sft.py --deepspeed config/ds_config.json

#  CUDA_VISIBLE_DEVICES=4,5 python3 trainer/train_sft.py