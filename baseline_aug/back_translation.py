import os
import json
import copy
import re
import torch
import numpy as np 
import random
from tqdm import tqdm
from baseline_aug.utils import *
def get_back_translation_data(fewaug,task_name,data_path,type='bt_lang_10'):
    dict_ans=torch.load('/workspace/zhoujing/FewGLUE_dev32/augmented/{}/dict_ans_google'.format(task_name))
    augmented_sentences=[]
    if type=='bt_lang_10':
        num=len(dict_ans)
        for (_,sentences) in dict_ans.items():
            augmented_sentences+=sentences
    else:
        num=6
        for keys in ['English','Spanish','French','German','Russian','Haitian Creole']:
            augmented_sentences+=dict_ans[keys]

    if task_name=='MultiRC':
        from pet.tasks import PROCESSORS, load_examples,TRAIN_SET
        examples=load_examples(task_name.lower(),os.path.join(data_path,task_name) , TRAIN_SET,num_examples=-1)
    else:
        examples=fewaug.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(task_name)))
    new_examples=[]
    elen=len(examples)
    if task_name.lower()=='copa':
        elen*=2
        for (i,e) in enumerate(examples):
            for n in range(num):
                tmp_e=copy.deepcopy(e)
                tmp_e['premise']=augmented_sentences[(elen*n+i)*3]
                tmp_e['choice1']=augmented_sentences[(elen*n+i)*3+1]
                tmp_e['choice2']=augmented_sentences[(elen*n+i)*3+2]
                new_examples.append(tmp_e)
    elif task_name.lower()=='multirc':
        elen=len(augmented_sentences)//num//3
        print(elen)
        for (i,e) in enumerate(examples):
            for n in range(num):
                tmp_e={"idx": e.meta['passage_idx'], "version": 1.1, \
                        "passage": {
                            "text": augmented_sentences[(elen*n+i)*3],
                            "questions": [{"question": augmented_sentences[(elen*n+i)*3+1],
                                "answers": [{"text": augmented_sentences[(elen*n+i)*3+2], "idx": e.meta['answer_idx'], "label": e.label}],
                                "idx": e.meta['question_idx']}]
                        }
                    }
                new_examples.append(tmp_e)
    elif task_name.lower()=='boolq':
        for (i,e) in enumerate(examples):
            for n in range(num):
                tmp_e=copy.deepcopy(e)
                tmp_e['passage']=augmented_sentences[(elen*n+i)*2]
                tmp_e['question']=augmented_sentences[(elen*n+i)*2+1]
                new_examples.append(tmp_e)
    elif task_name.lower()=='boolq':
        for (i,e) in enumerate(examples):
            for n in range(num):
                tmp_e=copy.deepcopy(e)
                tmp_e['passage']=augmented_sentences[(elen*n+i)*2]
                tmp_e['question']=augmented_sentences[(elen*n+i)*2+1]
                new_examples.append(tmp_e)
    elif task_name.lower()=='cb' or task_name.lower()=='rte':
        for (i,e) in enumerate(examples):
            for n in range(num):
                tmp_e=copy.deepcopy(e)
                tmp_e['premise']=augmented_sentences[(elen*n+i)*2]
                tmp_e['hypothesis']=augmented_sentences[(elen*n+i)*2+1]
                new_examples.append(tmp_e)
    fewaug.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(task_name,type)))
    return new_examples