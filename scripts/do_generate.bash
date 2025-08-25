export PYTHONPATH=$(pwd):$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python ./datagenerator/do_generate.py start_idx=0 end_idx=1
# CUDA_VISIBLE_DEVICES=2 python ./datagenerator/do_generate.py start_idx=35 end_idx=70
# CUDA_VISIBLE_DEVICES=3 python ./datagenerator/do_generate.py start_idx=70 end_idx=95
# CUDA_VISIBLE_DEVICES=4 python ./datagenerator/do_generate.py start_idx=95 end_idx=100