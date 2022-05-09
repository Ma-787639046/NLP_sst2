#!/bin/bash

MODEL_DIR_NAME=huggingface_bert_base_uncased_ori

python run.py --pretrain_model_dir_name=$MODEL_DIR_NAME \
                --run_custumized_task=sst2 \
                --repeated_times=8 \
                |& tee sst2.log
