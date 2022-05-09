#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
This script automatic calls run_glue.py multiple times, 
calulate mean & var of results, and save best finetune 
checkpoints.

@Time    :   2022/02/16 20:38:59
@Author  :   Ma (Ma787639046@outlook.com)
'''

import json
import csv
import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

class args:    
    default_argument_group = [f'\
    --do_train \
    --do_eval \
    --fp16 \
    --save_strategy=epoch \
    ']

    # Use sequential seeds
    seed_list = [i for i in range(1, 33)]    # 1 ~ 32

    eval_items = {
        'cola': ["eval_matthews_correlation"],
        'stsb': ["eval_pearson", "eval_spearmanr"],
        'rte': ["eval_accuracy"],
        'mrpc': ["eval_f1", "eval_accuracy"],
        'wnli': ["eval_accuracy"],
        'sst2': ["eval_accuracy"],
        'mnli': ["eval_accuracy"],
        'qnli': ["eval_accuracy"],
        'qqp': ["eval_accuracy", "eval_f1"]
    }

    huggingface_params = {
        'cola': {
            'task_name': 'cola',
            'learning_rate': '2e-5',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 32,
            'max_seq_length': 128
        },

        'stsb': {
            'task_name': 'stsb',
            'learning_rate': '2e-5',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 32,
            'max_seq_length': 128
        },

        'rte': {
            'task_name': 'rte',
            'learning_rate': '2e-5',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 32,
            'max_seq_length': 128
        },

        'mrpc': {
            'task_name': 'mrpc',
            'learning_rate': '2e-5',
            'num_train_epochs': 5,
            'per_device_train_batch_size': 32,
            'max_seq_length': 128
        },

        'wnli': {
            'task_name': 'wnli',
            'learning_rate': '2e-5',
            'num_train_epochs': 5,
            'per_device_train_batch_size': 32,
            'max_seq_length': 128
        }
    }

    # 注意，这只是用了rdrop推荐的finetune参数，实验本身没加rdrop loss
    rdrop_finetune_param = {
        'cola': {
            'task_name': 'cola',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 16,
            # 'max_steps': 5336,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'stsb': {
            'task_name': 'stsb',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 16,
            # 'max_steps': 3598,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'rte': {
            'task_name': 'rte',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 8,
            # 'max_steps': 2036,
            'num_train_epochs': 7,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'mrpc': {
            'task_name': 'mrpc',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 16,
            # 'max_steps': 2296,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'qqp': {
            'task_name': 'qqp',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            # 'max_steps': 113272,
            'num_train_epochs': 10,
            'warmup_ratio': 0.25,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'sst2': {
            'task_name': 'sst2',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            # 'max_steps': 20935,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'mnli': {
            'task_name': 'mnli',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            # 'max_steps': 123873,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'qnli': {
            'task_name': 'qnli',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            # 'max_steps': 33112,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'wnli': {
            'task_name': 'wnli',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 8,
            'num_train_epochs': 7,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        }
    }

    our_finetune_param = {
        'cola': {
            'task_name': 'cola',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 16,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'mnli': {
            'task_name': 'mnli',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            'num_train_epochs': 5,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'mrpc': {
            'task_name': 'mrpc',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 16,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'qnli': {
            'task_name': 'qnli',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            'num_train_epochs': 5,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'qqp': {
            'task_name': 'qqp',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            'num_train_epochs': 5,
            'warmup_ratio': 0.25,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'rte': {
            'task_name': 'rte',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 8,
            'num_train_epochs': 7,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'sst2': {
            'task_name': 'sst2',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 32,
            'num_train_epochs': 5,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        },
        'stsb': {
            'task_name': 'stsb',
            'learning_rate': '1e-5',
            'per_device_train_batch_size': 16,
            'num_train_epochs': 10,
            'warmup_ratio': 0.06,
            'adam_beta2': 0.98,
            'adam_epsilon': '1e-6',
            'evaluation_strategy': 'epoch',
            'weight_decay': 0.01,
            'lr_scheduler_type': 'polynomial'
        }
    }

def one_trail(gpu_id: int, task_name_group, temp_base_path, mode: str, rank_baseid: int, 
                glue_runner_path: str, model_path: str):
    if mode == 'single_gpu':
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} "
    elif mode == 'all_gpu':
        cmd = ""
    else: 
        raise NotImplementedError()
    cmd += f"python {glue_runner_path} "
    cmd += f" --model_name_or_path={model_path} "
    for arg_ in args.default_argument_group:
        cmd += arg_
    for item in task_name_group:
        cmd += f" --{item}={task_name_group[item]} "
    # Set a seed from seed list
    cmd += f" --seed={args.seed_list[rank_baseid + gpu_id]} "
    # cmd += f" --seed={np.random.randint(0, 1e2, 1)[0]} "
    # Make temp output path
    temp_dir = os.path.join(temp_base_path, task_name_group['task_name'] + f"_globalrank{rank_baseid + gpu_id}_" + '{0:%Y%m%d_%H%M%S}'.format(datetime.datetime.now()))
    os.makedirs(temp_dir, exist_ok=True)
    cmd += f"--output_dir {temp_dir}"
    os.system(cmd)

def run(task_name:str , task_name_group: dict, eval_group: list, repeated_times: int, 
        glue_runner_path: str, model_path: str, glue_result_path:str, mode: str):
    """ This will spawn task multiple times on GPU, 
        then return & save the mean & var of results, and save 
        best finetune checkpoints.  
    """
    temp_base_path = "./temp_" + '{0:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())    
    os.makedirs(temp_base_path)

    # Make final result output path
    final_result_path = os.path.join(glue_result_path, \
                                        task_name_group['task_name'])
    if torch.cuda.device_count() >= repeated_times:
        global_rank_lists = [repeated_times]
    else:
        global_rank_lists = []
        i = repeated_times
        while i > torch.cuda.device_count():
            global_rank_lists.append(torch.cuda.device_count())
            i -= torch.cuda.device_count()
        global_rank_lists.append(i)
    # rank_baseid records how many trails the script has finished, to call seed differently
    rank_baseid = 0
    for global_rank in global_rank_lists:
        mp.spawn(one_trail, args=(task_name_group, temp_base_path, mode, rank_baseid, glue_runner_path, model_path),\
             nprocs=global_rank)
        rank_baseid += global_rank
    
    # Summarize results
    results = {}
    for item in eval_group:
        results[item] = []
    
    max_eval_value = 0.0
    
    for dirname in os.listdir(temp_base_path):
        json_filepath = os.path.join(temp_base_path, dirname, 'eval_results.json')
        if os.path.exists(json_filepath):
            json_item = json.load(open(json_filepath))
            if json_item[eval_group[0]] > max_eval_value:
                max_eval_value = json_item[eval_group[0]]
                max_eval_dir = dirname  # 默认取eval_group中的第一个值
            for item in eval_group:
                results[item].append(json_item[item])
    
    # Save best model to output path
    model_output_path = os.path.join(glue_result_path, task_name_group['task_name'])
    if os.path.exists(model_output_path):
        os.system(f"rm -rf {model_output_path}")
    os.makedirs(model_output_path)
    os.system(f"cp {os.path.join(temp_base_path, max_eval_dir)}/* {model_output_path}")

    # Clean temp
    os.system(f"rm -rf {temp_base_path}")

    # Save all single value
    with open(os.path.join(glue_result_path, task_name + '_values.json'), 'a') as f:
        f.write(json.dumps(results))

    summary_filepath = os.path.join(glue_result_path, 'summary.csv')
    if not os.path.exists(summary_filepath):    # Write a title to summary.csv
        with open(summary_filepath, 'a') as f:
            csv_writer = csv.writer(f, dialect='excel')
            csv_writer.writerow(['Task', 'Date', 'Score'])
    # Save avg ± std to summary file
    with open(summary_filepath, 'a') as f:
        csv_writer = csv.writer(f, dialect='excel')
        str_value = ''
        for item in results:
            avg = np.average(results[item])
            avg = np.round(avg * 100, 2)
            std = np.std(results[item])
            std = np.round(std * 100, 2)
            str_value += f'{avg}±{std}/'
        str_value = str_value.strip('/')
        line = [task_name, '{0:%Y%m%d}'.format(datetime.datetime.now()), str_value]
        csv_writer.writerow(line)

    return results

def main():
    
    return

if __name__ == '__main__':
    os.chdir(os.path.split(os.path.realpath(__file__))[0])
    main()

