import os
import json
import shutil
from transformers import pipeline
from tqdm import tqdm
import torch.nn as nn

import argparse
# argparse模块是命令行选项、参数和子命令解析器

EXPERIMENT_ROOT_FOLDER = 'experiments'


parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=42, type=int,
                    help='Random seed for initialization')

# Model settings
parser.add_argument('--export_root', default='/Users/ansixu/final', type=str,
                    help='The pretrained directory where the model weights could be loaded')
parser.add_argument('--path', default='/Users/ansixu/final', type=str,
                    help='This path for data to feed into model')
parser.add_argument('--output_dir', default=None, type=str,
                    help='The output directory where the model checkpoints and predictions will be written')
parser.add_argument('--pretrained_dir', default=None, type=str,
                    help='The pretrained directory where the model weights could be loaded')


# Experiment settings
parser.add_argument('--do_train', action='store_true', help='Whether to run training')
parser.add_argument('--do_predict', action='store_true', help='Whether to run evaluation')
parser.add_argument('--train_both', action='store_true', help='Whether to train on both train data')
parser.add_argument('--train_source', action='store_true', help='Whether to only train on source data')
parser.add_argument('--train_target', action='store_true', help='Whether to only train on target data')
parser.add_argument('--val_size', default=0.1, type=float, help='Validation size from the dataset if not already split')
parser.add_argument('--test_size', default=0.2, type=float, help='Test size from the dataset if not already split')

parser.add_argument('--num_pi', default=3, type=int, help='Number of subtasks, should be at least 2')
parser.add_argument('--num_updates', default=4, type=int, help='Number of update steps for each subtask (inner update)')
parser.add_argument('--num_shots', default=20, type=int, help='Number of shots for each class in the query set')
parser.add_argument('--conf_threshold', default=0.8, type=float, help='Initial confidence threshold for pseudo labels')
parser.add_argument("--source_topic", default='topic1', type=str, help="Source topic data file for training.")
parser.add_argument("--source_data_type", default='gossipcop', type=str, help="Source file type for training. E.g., constraint")
parser.add_argument('--target_topic', default='topic3', type=str, help='Target topic data for transfer')
parser.add_argument('--target_data_type', default='', type=str, help='Source file type for training. E.g., constraint')
parser.add_argument('--load_model_path', default=None, type=str, help='Trained source model path for adaptation.')

# Optimization settings
parser.add_argument('--train_batchsize', default=4, type=int, help='Batch size used for training')
parser.add_argument('--eval_batchsize', default=24, type=int, help='Batch size used for evaluation')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='The source initial learning rate for optimizer')
parser.add_argument('--learning_rate_meta', default=1e-5, type=float, help='The meta initial learning rate for outer loop')
parser.add_argument('--lr_decay_meta', default=0.1, type=float, help='The minimum learning rate for outer loop')
parser.add_argument('--learning_rate_learner', default=0.1, type=float, help='The learner learning rate (Subtask)')
parser.add_argument('--learning_rate_lr', default=0.1, type=float, help='The (subtask) learning rate of learning rate')
parser.add_argument('--softmax_temp', default=0.01, type=float, help='The softmax temperature for learning')
parser.add_argument('--num_train_epochs', default=2, type=int, help='Total number of source training epochs to perform')
parser.add_argument('--num_iterations', default=20, type=int, help='Total number of adaptation iterations to perform')
parser.add_argument('--eval_interval', default=50, type=int, help='Validation interval in training')
parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Maximum gradient norm in training')

# Option for deleting model files
parser.add_argument('--del_model', default=False, type=bool, help='Whether to delete model after training')

args = parser.parse_args()