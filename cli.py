# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script can be used to train and evaluate either a regular supervised model or a PET/iPET model on
one of the supported tasks and datasets.
"""

import argparse
import copy
import os
from typing import Tuple
import shutil
import torch
import ast

from pet.tasks import PROCESSORS, load_examples, UNLABELED_SET, TRAIN_SET, DEV_SET, TEST_SET, METRICS, DEFAULT_METRICS
from pet.utils import eq_div, set_seed
from pet.wrapper import WRAPPER_TYPES, MODEL_CLASSES, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
import pet
import log

logger = log.get_logger('root')

import numpy as np

def load_pet_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for PET from the given command line arguments.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=args.wrapper_type, task_name=args.task_name, label_list=args.label_list,
                              max_seq_length=args.pet_max_seq_length, verbalizer_file=args.verbalizer_file,
                              cache_dir=args.cache_dir,
                              use_noisy_student=args.use_noisy_student, drop_prob=args.drop_prob,
                              fix_deberta=args.fix_deberta,
                              mixup=args.mixup,mixup_alpha=args.mixup_alpha)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.pet_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.pet_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.pet_num_train_epochs, max_steps=args.pet_max_steps,
                                gradient_accumulation_steps=args.pet_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, lm_training=args.lm_training, alpha=args.alpha)

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.pet_per_gpu_eval_batch_size,
                              decoding_strategy=args.decoding_strategy, priming=args.priming)

    return model_cfg, train_cfg, eval_cfg


def load_sequence_classifier_configs(args) -> Tuple[WrapperConfig, pet.TrainConfig, pet.EvalConfig]:
    """
    Load the model, training and evaluation configs for a regular sequence classifier from the given command line
    arguments. This classifier can either be used as a standalone model or as the final classifier for PET/iPET.
    """
    model_cfg = WrapperConfig(model_type=args.model_type, model_name_or_path=args.model_name_or_path,
                              wrapper_type=SEQUENCE_CLASSIFIER_WRAPPER, task_name=args.task_name,
                              label_list=args.label_list, max_seq_length=args.sc_max_seq_length,
                              verbalizer_file=args.verbalizer_file, cache_dir=args.cache_dir,
                              use_noisy_student=args.use_noisy_student, drop_prob=args.drop_prob,
                              fix_deberta=args.fix_deberta)

    train_cfg = pet.TrainConfig(device=args.device, per_gpu_train_batch_size=args.sc_per_gpu_train_batch_size,
                                per_gpu_unlabeled_batch_size=args.sc_per_gpu_unlabeled_batch_size, n_gpu=args.n_gpu,
                                num_train_epochs=args.sc_num_train_epochs, max_steps=args.sc_max_steps,
                                temperature=args.temperature,
                                gradient_accumulation_steps=args.sc_gradient_accumulation_steps,
                                weight_decay=args.weight_decay, learning_rate=args.learning_rate,
                                adam_epsilon=args.adam_epsilon, warmup_steps=args.warmup_steps,
                                max_grad_norm=args.max_grad_norm, use_logits=args.method != 'sequence_classifier')

    eval_cfg = pet.EvalConfig(device=args.device, n_gpu=args.n_gpu, metrics=args.metrics,
                              per_gpu_eval_batch_size=args.sc_per_gpu_eval_batch_size)

    return model_cfg, train_cfg, eval_cfg


def load_ipet_config(args) -> pet.IPetConfig:
    """
    Load the iPET config from the given command line arguments.
    """
    ipet_cfg = pet.IPetConfig(generations=args.ipet_generations, logits_percentage=args.ipet_logits_percentage,
                              scale_factor=args.ipet_scale_factor, n_most_likely=args.ipet_n_most_likely)
    return ipet_cfg


def main():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")

    # Required parameters
    parser.add_argument("--method", required=True, choices=['pet', 'ipet', 'sequence_classifier', 'noisy_student'],
                        help="The training method to use. Either regular sequence classification, PET or iPET.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True, choices=MODEL_CLASSES.keys(),
                        help="The type of the pretrained language model to use")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--task_name", default=None, type=str, required=True, choices=PROCESSORS.keys(),
                        help="The name of the task to train/evaluate on")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written")

    # PET-specific optional parameters
    parser.add_argument("--wrapper_type", default="mlm", choices=WRAPPER_TYPES,
                        help="The wrapper type. Set this to 'mlm' for a masked language model like BERT or to 'plm' "
                             "for a permuted language model like XLNet (only for PET)")
    parser.add_argument("--pattern_ids", default=[0], type=int, nargs='+',
                        help="The ids of the PVPs to be used (only for PET)")
    parser.add_argument("--lm_training", action='store_true',
                        help="Whether to use language modeling as auxiliary task (only for PET)")
    parser.add_argument("--alpha", default=0.9999, type=float,
                        help="Weighting term for the auxiliary language modeling task (only for PET)")
    parser.add_argument("--temperature", default=2, type=float,
                        help="Temperature used for combining PVPs (only for PET)")
    parser.add_argument("--verbalizer_file", default=None,
                        help="The path to a file to override default verbalizers (only for PET)")
    parser.add_argument("--reduction", default='mean', choices=['wmean', 'mean'],
                        help="Reduction strategy for merging predictions from multiple PET models. Select either "
                             "uniform weighting (mean) or weighting based on train set accuracy (wmean)")
    parser.add_argument("--decoding_strategy", default='default', choices=['default', 'ltr', 'parallel'],
                        help="The decoding strategy for PET with multiple masks (only for PET)")
    parser.add_argument("--no_distillation", action='store_true',
                        help="If set to true, no distillation is performed (only for PET)")
    parser.add_argument("--pet_repetitions", default=3, type=int,
                        help="The number of times to repeat PET training and testing with different seeds.")
    parser.add_argument("--pet_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for PET. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--pet_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for PET training.")
    parser.add_argument("--pet_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for PET evaluation.")
    parser.add_argument("--pet_per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for auxiliary language modeling examples in PET.")
    parser.add_argument('--pet_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass in PET.")
    parser.add_argument("--pet_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform in PET.")
    parser.add_argument("--pet_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform in PET. Override num_train_epochs.")

    # SequenceClassifier-specific optional parameters (also used for the final PET classifier)
    parser.add_argument("--sc_repetitions", default=1, type=int,
                        help="The number of times to repeat seq. classifier training and testing with different seeds.")
    parser.add_argument("--sc_max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization for sequence classification. "
                             "Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--sc_per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for sequence classifier training.")
    parser.add_argument("--sc_per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for sequence classifier evaluation.")
    parser.add_argument("--sc_per_gpu_unlabeled_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for unlabeled examples used for distillation.")
    parser.add_argument('--sc_gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass for "
                             "sequence classifier training.")
    parser.add_argument("--sc_num_train_epochs", default=3, type=float,
                        help="Total number of training epochs to perform for sequence classifier training.")
    parser.add_argument("--sc_max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform for sequence classifier training. "
                             "Override num_train_epochs.")

    # iPET-specific optional parameters
    parser.add_argument("--ipet_generations", default=3, type=int,
                        help="The number of generations to train (only for iPET)")
    parser.add_argument("--ipet_logits_percentage", default=0.25, type=float,
                        help="The percentage of models to choose for annotating new training sets (only for iPET)")
    parser.add_argument("--ipet_scale_factor", default=5, type=float,
                        help="The factor by which to increase the training set size per generation (only for iPET)")
    parser.add_argument("--ipet_n_most_likely", default=-1, type=int,
                        help="If >0, in the first generation the n_most_likely examples per label are chosen even "
                             "if their predicted label is different (only for iPET)")

    # Other optional parameters
    parser.add_argument("--train_examples", default=-1, type=int,
                        help="The total number of train examples to use, where -1 equals all examples.")
    parser.add_argument("--test_examples", default=-1, type=int,
                        help="The total number of test examples to use, where -1 equals all examples.")
    parser.add_argument("--unlabeled_examples", default=-1, type=int,
                        help="The total number of unlabeled examples to use, where -1 equals all examples")
    parser.add_argument("--split_examples_evenly", action='store_true',
                        help="If true, train examples are not chosen randomly, but split evenly across all labels.")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where to store the pre-trained models downloaded from S3.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--do_train', action='store_true',
                        help="Whether to perform training")
    parser.add_argument('--do_eval', action='store_true',
                        help="Whether to perform evaluation")
    parser.add_argument('--priming', action='store_true',
                        help="Whether to use priming for evaluation")
    parser.add_argument("--eval_set", choices=['dev', 'test'], default='dev',
                        help="Whether to perform evaluation on the dev set or the test set")

    parser.add_argument("--search_type",default='',type=str)
    parser.add_argument("--aug_ids", default=[0,2], type=int, nargs='+',)
    parser.add_argument("--filter_pattern",default=-1,type=int)
    parser.add_argument("--fixla_ratio",default='[[-1,-1],[-1,-1]]',type=str)
    parser.add_argument("--fixla_num",default='[[14,14],[18,18]]',type=str)
    parser.add_argument("--rmdup_num",default=1,type=int)


    # TODO: noisystudent
    parser.add_argument("--use_noisy_student", action="store_true", help="Whether to use noisy student.")
    parser.add_argument("--drop_prob", default=1.0, type=float, help="Dropout probability for noising input data.")
    parser.add_argument("--t5_augment_file_path", help="t5_augment data as unlabeled data for noisy student.")
    # t5_flip_0.5_rand_iter_10_sample1_beam1_augnum10_train.jsonl

    parser.add_argument("--sampler_seeds", type=list, default=[10, 20, 30])
    parser.add_argument("--fix_deberta", action="store_true")


    args = parser.parse_args()
    logger.info("Parameters: {}".format(args))
    
    assert not ('flip' in args.search_type and 'max_prevla' in args.search_type) and not ('keep' in args.search_type and 'max_otherla' in args.search_type)
    set_seed(args.seed)
    if 'topk' in args.search_type:
        args.output_dir=os.path.join(args.output_dir,args.fixla_num)
    if 'topp' in args.search_type:
        args.output_dir=os.path.join(args.output_dir,args.fixla_ratio)
    if 'rmdup' in args.search_type:
        args.output_dir=os.path.join(args.output_dir,'rmdup{}'.format(args.rmdup_num))
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) \
           and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        # shutil.rmtree(args.output_dir)
    

    if args.search_type.startswith('mixup'): 
        args.mixup=True
        if '_' in args.search_type:
            args.mixup_alpha=float(args.search_type.split('_')[1])
        else:
            args.mixup_alpha=0.5
    else:
        args.mixup=False; args.mixup_alpha=-1

    # Setup CUDA, GPU & distributed training
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    args.n_gpu = torch.cuda.device_count()

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in PROCESSORS:
        raise ValueError("Task '{}' not found".format(args.task_name))
    processor = PROCESSORS[args.task_name]()
    args.label_list = processor.get_labels()

    train_ex_per_label, test_ex_per_label = None, None
    train_ex, test_ex = args.train_examples, args.test_examples
    if args.split_examples_evenly:
        train_ex_per_label = eq_div(args.train_examples, len(args.label_list)) if args.train_examples != -1 else -1
        test_ex_per_label = eq_div(args.test_examples, len(args.label_list)) if args.test_examples != -1 else -1
        train_ex, test_ex = None, None

    eval_set = TEST_SET if args.eval_set == 'test' else DEV_SET

    train_data = load_examples(
        args.task_name, args.data_dir, TRAIN_SET, num_examples=train_ex, num_examples_per_label=train_ex_per_label)
    eval_data = load_examples(
        args.task_name, args.data_dir, eval_set, num_examples=test_ex, num_examples_per_label=test_ex_per_label)

    # TODO: noisy student

    if args.use_noisy_student:
        taskname_map = {'copa': 'COPA', 'rte': 'RTE', 'boolq': "BoolQ", 'multirc': "MultiRC", 'wic': 'WiC', 'wsc': "WSC", 'cb': 'CB'}
        t5_augment_file_paths=eval(args.t5_augment_file_path)
        unlabeled_data=[]
        for t5_augment_file_path in t5_augment_file_paths:
            unlabeled_data += processor._create_examples(
                    os.path.join('/'.join(args.data_dir.split('/')[:-1]), "augmented/{}/{}".format(taskname_map[args.task_name], t5_augment_file_path+'_train.jsonl')), "t5_augment_as_unlabeled")
        logger.info("t5_augment_as_unlabeled num_examples: {}".format(len(unlabeled_data)))

    else:
        unlabeled_data = load_examples(
            args.task_name, args.data_dir, UNLABELED_SET, num_examples=args.unlabeled_examples)

    args.metrics = METRICS.get(args.task_name, DEFAULT_METRICS)

    pet_model_cfg, pet_train_cfg, pet_eval_cfg = load_pet_configs(args)
    sc_model_cfg, sc_train_cfg, sc_eval_cfg = load_sequence_classifier_configs(args)
    ipet_cfg = load_ipet_config(args)


    if args.search_type=='baseline' or args.search_type.startswith('mixup'):
        pass
    elif args.search_type.startswith('genaug'):
        taskname_map={'copa':'COPA','rte':'RTE','boolq':"BoolQ",'multirc':"MultiRC",'wic':'WiC','wsc':"WSC",'cb':'CB'}
        new_examples=processor._create_examples(os.path.join('/'.join(args.data_dir.split('/')[:-1]), "augmented/{}/{}_train.jsonl".format(taskname_map[args.task_name],'_'.join(''.join(args.search_type.split('_filter_')[0]).split('_')[1:]))),"train")
    else:
        taskname_map={'rte':'RTE','boolq':'BoolQ','cb':'CB','multirc':'MultiRC','copa':'COPA','wic':'WiC','wsc':'WSC'}
        new_examples=processor._create_examples(os.path.join('/'.join(args.data_dir.split('/')[:-1]), "augmented/{}/{}_train.jsonl".format(taskname_map[args.task_name],args.search_type.split('_filter_')[0])), "train")

    if 'filter' in args.search_type:
        pattern_iter_output_dir='results/baseline/pet/{}_{}_model/'.format(args.task_name.lower(), args.model_type)
        from genaug import confidence_filter
        # if args.filter_pattern==-1:
        #     # best_pattern_map={'boolq':0,'rte':3,'cb':3,'multirc':1,'wic':1,'wsc':2,'copa':0}
        #     # best_iter_map={'boolq':0,'rte':1,'cb':1,'multirc':2,'wic':2,'wsc':2,'copa':0}
        #     best_pattern_map={'boolq':1,'rte':3,'cb':2,'multirc':0,'wic':1,'wsc':2,'copa':1}
        #     best_iter_map={'boolq':2,'rte':1,'cb':2,'multirc':0,'wic':2,'wsc':0,'copa':2}
        #     args.filter_pattern=best_pattern_map[args.task_name.lower()]
        #     args.best_iter_pattern=best_iter_map[args.task_name.lower()]
        if args.filter_pattern==-1:
            subdirs=next(os.walk(pattern_iter_output_dir))[1]
            best_score=-100
            for subdir in subdirs:
                results_file=os.path.join(pattern_iter_output_dir,subdir,'results.json')
                with open(results_file,'r') as fh:
                    results=ast.literal_eval(fh.read().lower().replace('nan','100'))
                    score=np.mean([y for (x,y) in results['test_set_after_training'].items()])
                    if score>best_score:
                        best_score=score
                        new_pattern_iter_output_dir=os.path.join(pattern_iter_output_dir,subdir)
            pattern_iter_output_dir=new_pattern_iter_output_dir
        else:
            pattern_iter_output_dir=os.path.join(pattern_iter_output_dir,'p{}-i{}'.format(args.filter_pattern,0))
        myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir=pattern_iter_output_dir)
        if 'flip' in args.search_type and 'max_otherla' not in args.search_type and args.task_name.lower()!='wsc':
            keep_path=os.path.join('/'.join(args.data_dir.split('/')[:-1]), "augmented/{}/{}_train.jsonl".format(taskname_map[args.task_name],'_'.join(''.join(args.search_type.replace('flip','keep').split('_filter_')[0]).split('_')[1:])))
            if os.path.exists(keep_path)==True:
                keep_examples=processor._create_examples(keep_path,"train")
                examples=new_examples+keep_examples
            else:
                examples=new_examples
        elif 'keep' in args.search_type and 'max_prevla' not in args.search_type and args.task_name.lower()!='wsc':
            keep_path=os.path.join('/'.join(args.data_dir.split('/')[:-1]), "augmented/{}/{}_train.jsonl".format(taskname_map[args.task_name],'_'.join(''.join(args.search_type.replace('keep','flip').split('_filter_')[0]).split('_')[1:])))
            if os.path.exists(keep_path)==True:
                keep_examples=processor._create_examples(keep_path,"train")
                examples=new_examples+keep_examples
            else:
                examples=new_examples
        else:
            examples=new_examples

        new_examples,filtered_num=myfilter.recover_labels(myfilter.wrapper,examples,pet_eval_cfg,recover_type=args.search_type.split('_filter_')[1],fixla_ratio=eval(args.fixla_ratio),fixla_num=eval(args.fixla_num),rmdup_num=args.rmdup_num)
        myfilter.del_finetuned_model()          

    if args.search_type=='baseline' or args.search_type.startswith('mixup'):
        pass
    elif (args.search_type.startswith('eda') or args.search_type.startswith('bt')) and "filter" not in args.search_type:
        train_data = train_data + new_examples
    else:
        if 'max_eachla' in args.search_type:
            train_data=train_data+new_examples
        else:
            train_data=train_data*max(1,int(len(new_examples)//len(train_data)))+new_examples

    if args.drop_prob!=1:
        pet_model_cfg.use_noisy_student=True
        
    if args.method == 'pet':
        results=pet.train_pet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                      pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                      ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                      reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                      eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval,
                      no_distillation=args.no_distillation, seed=args.seed, sampler_seeds=args.sampler_seeds)

        mean_result={}
        for result in results:
            for metric,res in result.items():
                if metric not in mean_result: mean_result[metric]=[]
                mean_result[metric].append(res)
        for (x,y) in mean_result.items():
            mean_result[x]=np.mean(y)

        template_name=['task_name','search_type']
        template_values=[args.task_name,args.search_type]
        if 'global_topk' in args.search_type:
            template_name.append('fixla_num')
            template_values.append(args.fixla_num)
        elif 'global_topp' in args.search_type:
            template_name.append('fixla_ratio')
            template_values.append(args.fixla_ratio)
        if 'rmdup' in args.search_type:
            template_name.append('rmdup_num')
            template_values.append(args.rmdup_num)
        if 'filter' in args.search_type:
            template_name.append('filtered_num')
            template_values.append(filtered_num)
        template_name.append('result')
        template_name.append('mean_result')
        if args.pet_repetitions!=1:
            if args.search_type.startswith('genaug'):
                writer=open(os.path.join('results/','pet_total_genaug_rep{}_{}.csv'.format(args.pet_repetitions,args.task_name)),'a+')
            else:
                writer=open(os.path.join('results/','pet_total_rep{}_{}.csv'.format(args.pet_repetitions,args.task_name)),'a+')
        else:
            if args.search_type.startswith('genaug'):
                writer=open(os.path.join('results/','pet_total_genaug_{}.csv'.format(args.task_name)),'a+')
            else:
                writer=open(os.path.join('results/','pet_total_{}.csv'.format(args.task_name)),'a+')
        writer.write((': {}, '.join(template_name)+': {}\n').format(*template_values+[results]+[mean_result]))
        writer.close()

    elif args.method.startswith('noisy_student'):
        results=pet.train_noisy_student(pet_model_cfg, pet_train_cfg, pet_eval_cfg, ipet_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                       pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                       ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                       reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed, sampler_seeds=args.sampler_seeds,
                       fixla_ratio=args.fixla_ratio)

        mean_result={}
        for result in results:
            for metric,res in result.items():
                if metric not in mean_result: mean_result[metric]=[]
                mean_result[metric].append(res)
        for (x,y) in mean_result.items():
            mean_result[x]=np.mean(y)

        template_name=['task_name','search_type','augmented_file','drop_prob','fixla_ratio','filtered_num','result','mean_result']
        template_values=[args.task_name,args.search_type,args.t5_augment_file_path,args.drop_prob,args.fixla_ratio,filtered_num]
        writer=open(os.path.join('results/','pet_total_noisy_{}.csv'.format(args.task_name)),'a+')
        writer.write((': {}, '.join(template_name)+': {}\n').format(*template_values+[results]+[mean_result]))
        writer.close()        
    elif args.method == 'ipet':
        pet.train_ipet(pet_model_cfg, pet_train_cfg, pet_eval_cfg, ipet_cfg, sc_model_cfg, sc_train_cfg, sc_eval_cfg,
                       pattern_ids=args.pattern_ids, output_dir=args.output_dir,
                       ensemble_repetitions=args.pet_repetitions, final_repetitions=args.sc_repetitions,
                       reduction=args.reduction, train_data=train_data, unlabeled_data=unlabeled_data,
                       eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed, sampler_seeds=args.sampler_seeds)

    elif args.method == 'sequence_classifier':
        pet.train_classifier(sc_model_cfg, sc_train_cfg, sc_eval_cfg, output_dir=args.output_dir,
                             repetitions=args.sc_repetitions, train_data=train_data, unlabeled_data=unlabeled_data,
                             eval_data=eval_data, do_train=args.do_train, do_eval=args.do_eval, seed=args.seed, sampler_seeds=args.sampler_seeds)

    else:
        raise ValueError(f"Training method '{args.method}' not implemented")


if __name__ == "__main__":
    main()
