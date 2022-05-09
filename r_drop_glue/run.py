#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''


@Time    :   2022/02/17 00:29:41
@Author  :   Ma (Ma787639046@outlook.com)
'''

import os
import argparse

os.chdir(os.path.split(os.path.realpath(__file__))[0])
from glue_batch_caller import run, args as runner_args

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--working_dir",
                        default="/mmu_nlp/wuxing/maguangyuan/clue_go/bert_macl",
                        type=str,
                        help="bert_macl working dir")
    parser.add_argument("--pretrain_model_dir_name",
                        default=None,
                        type=str,
                        required=True,
                        help="the name of pretrained model dir in `results` dir")
    parser.add_argument("--task_list_group",
                        default=None,
                        type=str,
                        choices=['small', 'large1', 'large2', None],
                        help="")
    parser.add_argument("--run_custumized_task",
                        default=None,
                        type=str,
                        choices=['rte', 'cola', 'stsb', 'mrpc', 'sst2', 'qnli', 'mnli', 'qqp'],
                        help="If `task_list_group` is None, then will run task of run_custumized_task")
    parser.add_argument("--repeated_times",
                        default=8,
                        required=True,
                        type=int,
                        help="")

    input_args = parser.parse_args()

    # working_dir = "/mmu_nlp/wuxing/maguangyuan/clue_go/bert_macl"
    glue_runner_path = os.path.join(input_args.working_dir, "glue/r_drop_glue/run_glue.py")

    # Name or relative path of pretrained model's output dir 
    pretrain_output_dir = os.path.join(input_args.working_dir, "results", input_args.pretrain_model_dir_name)
    model_path = os.path.join(pretrain_output_dir, 'model')
    glue_result_path = os.path.join(pretrain_output_dir, 'glue_results')

    if input_args.task_list_group == None:
        assert input_args.run_custumized_task != None, \
        "Please set `task_list_group` or your customized task in `run_custumized_task`"
        task_list = list()
        task_list.append(input_args.run_custumized_task)
    elif input_args.task_list_group == 'small':
        task_list = ['rte', 'cola', 'stsb', 'mrpc']
    elif input_args.task_list_group == 'large1':
        task_list = ['sst2', 'qnli', 'mnli']
    elif input_args.task_list_group == 'large2':
        task_list = ['qqp']
    else: raise NotImplementedError()

    for task_name in task_list:
        print(f"Running {task_name}...")
        run(task_name=task_name, 
            task_name_group=runner_args.rdrop_finetune_param[task_name],
            eval_group=runner_args.eval_items[task_name], 
            repeated_times=input_args.repeated_times, 
            glue_runner_path=glue_runner_path,
            model_path=model_path, 
            glue_result_path=glue_result_path, 
            mode='single_gpu')
    return

if __name__ == '__main__':
    main()

