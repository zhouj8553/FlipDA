# import sys
# import importlib
# importlib.reload(sys)
# import codecs
# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

import os
import json
import copy
import re
import torch
import numpy as np 
import random
from tqdm import tqdm
from baseline_aug.utils import *

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

class FewGLUEAug():
    def __init__(self,):
        pass

    def read_jsonl(self,file_path):
        examples=[]
        with open(file_path,"r",encoding="utf8") as f:
            for line in f:
                example_json=json.loads(line)
                # print(example_json)
                examples.append(example_json)
        return examples
        
    def save_jsonl(self,examples,save_path):
        with open(save_path,"w",encoding="utf8") as f:
            for e in examples:
                f.write(json.dumps(e)+'\n')
        f.close()

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in examples:
            for _ in range(aug_num):
                new_premise=aug_func(e["premise"],**aug_kwargs)
                new_hypothesis=aug_func(e["hypothesis"],**aug_kwargs)
                if isinstance(new_premise,list):
                    for (x,y) in zip(new_premise,new_hypothesis):
                        tmp_e=copy.deepcopy(e)
                        tmp_e["premise"]=x 
                        tmp_e["hypothesis"]=y 
                        new_examples.append(tmp_e)
                else:
                    tmp_e=copy.deepcopy(e)
                    tmp_e["premise"]=new_premise
                    tmp_e["hypothesis"]=new_hypothesis
                    new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        # import pdb
        # pdb.set_trace()
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

class RTEAug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug,self).__init__()
        self.TASK_NAME="RTE"

class CBAug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug,self).__init__()
        self.TASK_NAME="CB"

class BoolQAug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug,self).__init__()
        self.TASK_NAME="BoolQ"

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in examples:
            for _ in range(aug_num):
                new_premise=aug_func(e["question"],**aug_kwargs)
                new_hypothesis=aug_func(e["passage"],**aug_kwargs)
                if isinstance(new_premise,list):
                    for (x,y) in zip(new_premise,new_hypothesis):
                        tmp_e=copy.deepcopy(e)
                        tmp_e["question"]=x 
                        tmp_e["passage"]=y 
                        new_examples.append(tmp_e)
                else:
                    tmp_e=copy.deepcopy(e)
                    tmp_e["question"]=new_premise
                    tmp_e["passage"]=new_hypothesis
                    new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

class COPAAug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug,self).__init__()
        self.TASK_NAME="COPA"

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in examples:
            for _ in range(aug_num):
                # import pdb
                # pdb.set_trace()
                new_premise=aug_func(e["premise"],**aug_kwargs)
                new_choice1=aug_func(e["choice1"],**aug_kwargs)
                new_choice2=aug_func(e["choice2"],**aug_kwargs)
                if isinstance(new_premise,list):
                    for (x,y,z) in zip(new_premise,new_choice1,new_choice2):
                        tmp_e=copy.deepcopy(e)
                        tmp_e["premise"]=x 
                        tmp_e["choice1"]=y 
                        tmp_e["choice2"]=z 
                        new_examples.append(tmp_e)
                else:
                    tmp_e=copy.deepcopy(e)
                    tmp_e["premise"]=new_premise
                    tmp_e["choice1"]=new_choice1
                    tmp_e["choice2"]=new_choice2
                    new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples


class WiCAug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug,self).__init__()
        self.TASK_NAME="WiC"

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in examples:
            seq1=whitepiece_line_tokenizer(e["sentence1"])
            seq2=whitepiece_line_tokenizer(e["sentence2"])
            word=e["word"]
            idx1,idx2=seq1.index(e["sentence1"][e["start1"]:e["end1"]]),seq2.index(e["sentence2"][e["start2"]:e["end2"]])
            sentences_to_be_augmented=[]
            sentences_to_be_augmented.append(" ".join(seq1[:idx1]))
            sentences_to_be_augmented.append(" ".join(seq1[idx1+1:]))
            sentences_to_be_augmented.append(" ".join(seq2[:idx2]))
            sentences_to_be_augmented.append(" ".join(seq2[idx2+1:]))
            for _ in range(aug_num):
                total_sentences=[]
                for x in sentences_to_be_augmented:
                    my_re = re.compile(r"[A-Za-z]",re.S)
                    res = re.findall(my_re,x)
                    # if len(res)==0:
                    #     total_sentences.append(x)
                    # else:
                    total_sentences.append(aug_func(x,**aug_kwargs))
                if isinstance(total_sentences[0],list):
                    for (a,b,c,d) in zip(total_sentences[0],total_sentences[1],total_sentences[2],total_sentences[3]):
                        tmp_e=copy.deepcopy(e)
                        tmp_e["sentence1"]=" ".join(a.split()+[e["sentence1"][e["start1"]:e["end1"]]]+b.split())
                        tmp_e["sentence2"]=" ".join(c.split()+[e["sentence2"][e["start2"]:e["end2"]]]+d.split())
                        tmp_e["start1"]=len(a)+1
                        tmp_e["start2"]=len(c)+1
                        tmp_e["end1"]=len(a)+e["end1"]-e["start1"]+1
                        tmp_e["end2"]=len(c)+e["end2"]-e["start2"]+1
                        new_examples.append(tmp_e)
                else:
                    a,b,c,d=total_sentences[0],total_sentences[1],total_sentences[2],total_sentences[3]
                    tmp_e=copy.deepcopy(e)
                    tmp_e["sentence1"]=" ".join(a.split()+[e["sentence1"][e["start1"]:e["end1"]]]+b.split())
                    tmp_e["sentence2"]=" ".join(c.split()+[e["sentence2"][e["start2"]:e["end2"]]]+d.split())
                    tmp_e["start1"]=len(a)+1
                    tmp_e["start2"]=len(c)+1
                    tmp_e["end1"]=len(a)+e["end1"]-e["start1"]+1
                    tmp_e["end2"]=len(c)+e["end2"]-e["start2"]+1
                    new_examples.append(tmp_e)

        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples


class WSCAug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug,self).__init__()
        self.TASK_NAME="WSC"

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in examples:
            seq=whitepiece_line_tokenizer(e["text"])
            idx1,idx2=e["target"]["span1_index"],e["target"]["span2_index"]
            if idx1>idx2:
                fill_text1=e["target"]["span2_text"].split();fill_text2=e["target"]["span1_text"].split()
                idx2,idx1=idx1,idx2
                swap=True
            else:
                fill_text1=e["target"]["span1_text"].split();fill_text2=e["target"]["span2_text"].split()
                swap=False
            sentences_to_be_augmented=[]
            sentences_to_be_augmented.append(" ".join(seq[:idx1]))
            sentences_to_be_augmented.append(" ".join(seq[idx1+len(fill_text1):idx2]))
            sentences_to_be_augmented.append(" ".join(seq[idx2+len(fill_text2):]))
            for _ in range(aug_num):
                total_sentences=[]
                for x in sentences_to_be_augmented:
                    total_sentences.append(aug_func(x,**aug_kwargs))
                if isinstance(total_sentences[0],list):
                    for (text1,text2,text3) in zip(total_sentences[0],total_sentences[1],total_sentences[2]):
                        tmp_e=copy.deepcopy(e)
                        if swap==True:
                            tmp_e["target"]["span2_index"]=len(text1.split())
                            tmp_e["target"]["span1_index"]=len((" ".join(text1.split()+[e["target"]["span2_text"]]+text2.split())).split())
                            tmp_e["text"]=" ".join(text1.split()+[e["target"]["span2_text"]]+text2.split()+[e["target"]["span1_text"]]+text3.split())
                        else:
                            tmp_e["target"]["span1_index"]=len(text1.split())
                            tmp_e["target"]["span2_index"]=len((" ".join(text1.split()+[e["target"]["span1_text"]]+text2.split())).split())
                            tmp_e["text"]=" ".join(text1.split()+[e["target"]["span1_text"]]+text2.split()+[e["target"]["span2_text"]]+text3.split())
                        new_examples.append(tmp_e)
                else:
                    text1,text2,text3=total_sentences[0],total_sentences[1],total_sentences[2]
                    tmp_e=copy.deepcopy(e)
                    if swap==True:
                        tmp_e["target"]["span2_index"]=len(text1.split())
                        tmp_e["target"]["span1_index"]=len((" ".join(text1.split()+[e["target"]["span2_text"]]+text2.split())).split())
                        tmp_e["text"]=" ".join(text1.split()+[e["target"]["span2_text"]]+text2.split()+[e["target"]["span1_text"]]+text3.split())
                    else:
                        tmp_e["target"]["span1_index"]=len(text1.split())
                        tmp_e["target"]["span2_index"]=len((" ".join(text1.split()+[e["target"]["span1_text"]]+text2.split())).split())
                        tmp_e["text"]=" ".join(text1.split()+[e["target"]["span1_text"]]+text2.split()+[e["target"]["span2_text"]]+text3.split())
                    new_examples.append(tmp_e)

        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples
        

class MultiRCAug(FewGLUEAug):
    def __init__(self):
        super(FewGLUEAug,self).__init__()
        self.TASK_NAME="MultiRC"

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl").format(self.TASK_NAME))
        new_examples=[]
        for e in examples:
            for _ in range(aug_num):
                # import pdb
                # pdb.set_trace()
                passage=aug_func(e["passage"]["text"],**aug_kwargs)
                question=aug_func(e["passage"]["questions"][0]["question"],**aug_kwargs)
                answers=[x["text"] for x in e["passage"]["questions"][0]["answers"]]
                # print(e)
                # aug_answers=[aug_func(x,**aug_kwargs) for x in answers]
                aug_answers=answers
                if isinstance(passage,list):
                    if not isinstance(aug_answers[0],list):
                        aug_answers=[[x] for x in aug_answers]
                    for (i,(x,y)) in enumerate(zip(passage,question)):
                        tmp_e=copy.deepcopy(e)
                        tmp_e["passage"]["text"]=x 
                        tmp_e["passage"]["questions"][0]["question"]=y
                        for (j,z) in enumerate(aug_answers):
                            if i>=len(z):
                                 tmp_e["passage"]["questions"][0]["answers"][j]["text"]=z[0]
                            else:
                                tmp_e["passage"]["questions"][0]["answers"][j]["text"]=z[i]
                        # print(tmp_e)
                        new_examples.append(tmp_e)
                else:
                    tmp_e=copy.deepcopy(e)
                    tmp_e["passage"]["text"]=passage
                    tmp_e["passage"]["questions"][0]["question"]=question
                    for (i,zz) in enumerate(aug_answers): 
                        tmp_e["passage"]["questions"][0]["answers"][i]["text"]=zz 
                    new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

import argparse
def init_args():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--aug_method",required=True,type=str)
    args = parser.parse_args()
    return args

FEWGLUEAUGS = {
    "wsc": WSCAug,
    "rte": RTEAug,
    "cb": CBAug,
    "wic": WiCAug,
    "boolq": BoolQAug,
    "copa": COPAAug,
    "multirc": MultiRCAug,
} 


def main():
    args=init_args()
    data_path="/workspace/zhoujing/FewGLUE_dev32/"
    aug_kwargs={}
    set_seed(1)
    aug_num=1
    if args.aug_method.startswith('bt'):
        from baseline_aug import back_translation
        fewaug=FewGLUEAug()
        back_translation.get_back_translation_data(fewaug,args.task_name,data_path,type=args.aug_method)
        return
    myaug=FEWGLUEAUGS[args.task_name.lower()]()
    if args.aug_method=="eda":
        from baseline_aug import eda
        aug_func=eda.eda
        aug_func_name="eda"
    elif args.aug_method=="eda_punc":
        from baseline_aug import eda_punc
        aug_func=eda_punc.eda
        aug_func_name="eda_punc"
    elif args.aug_method.startswith("knn"):
        from baseline_aug import knn_replacement
        knn_aug=knn_replacement.KNNAugment()
        aug_func=knn_aug.augment
        aug_func_name=args.aug_method
        aug_kwargs={"M":int(args.aug_method.split("_")[1]),"p":float(args.aug_method.split("_")[2])}
        aug_num=10
    elif args.aug_method.startswith("synonym"):
        from baseline_aug import eda
        aug_func=eda.sr_replacement
        aug_func_name=args.aug_method
        aug_kwargs={"alpha_sr":float(args.aug_method.split("_")[1])}
        aug_num=1
    elif args.aug_method.startswith("mlm"):
        from baseline_aug import MLM_replacement
        mlm_aug=MLM_replacement.MLMAugment()
        aug_func=mlm_aug.augment
        aug_func_name='mlm'
        aug_kwargs={"p":float(args.aug_method.split("_")[1])}
        aug_num=10
    elif args.aug_method.startswith("tinybert"):
        from baseline_aug import tinybert
        tinybert_aug=tinybert.Augment()
        aug_func=tinybert_aug.augment_sentences
        aug_func_name='tinybert'
        aug_kwargs={}
        aug_num=1
    elif args.aug_method.startswith("t5_mlm"):
        from baseline_aug import T5_MLM_replacement
        mlm_aug=T5_MLM_replacement.MLMAugment()
        aug_func=mlm_aug.augment
        aug_func_name='t5_mlm_0.1'
        aug_kwargs={"p":float(args.aug_method.split("_")[2])}
        aug_num=10
    new_examples=myaug.augment(data_path,aug_func,aug_func_name,aug_kwargs,aug_num=aug_num)


if __name__ == "__main__":
    main()
"""
from baseline_aug import total_aug
import imp
imp.reload(total_aug)
myaug=total_aug.COPAAug()

# myaug=total_aug.RTEAug()
# myaug=total_aug.BoolQAug()
# myaug=total_aug.CBAug()
# myaug=total_aug.WiCAug()
# myaug=total_aug.MultiRCAug()
myaug=total_aug.WSCAug()
data_path="/workspace/zhoujing/FewGLUE_dev32/"
# from baseline_aug import eda_punc
# aug_func=eda_punc.eda
# aug_func_name="eda_punc"
# aug_kwargs={}
from baseline_aug import tinybert
import imp
imp.reload(tinybert)
tinybert_aug=tinybert.Augment()
aug_func=tinybert_aug.augment_sentences
aug_func_name='tinybert'
aug_kwargs={}
new_examples=myaug.augment(data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1)
"""

# import json
# examples=[]
# with open("/workspace/zhoujing/FewGLUE_dev32/RTE/train.jsonl","r",encoding="utf8") as f:
#     for line in f:
#         example_json=json.loads(line)
#         print(example_json)
#         examples.append(example_json)

"""
nohup python -m baseline_aug.total_aug --task_name copa --aug_method eda >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name boolq --aug_method eda >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name rte --aug_method eda >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wsc --aug_method eda >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wic --aug_method eda >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name multirc --aug_method eda >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name cb --aug_method eda >myout.file 2>&1 &

nohup python -m baseline_aug.total_aug --task_name copa --aug_method eda_punc >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name boolq --aug_method eda_punc >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name rte --aug_method eda_punc >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wsc --aug_method eda_punc >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wic --aug_method eda_punc >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name multirc --aug_method eda_punc >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name cb --aug_method eda_punc >myout.file 2>&1 &


nohup python -m baseline_aug.total_aug --task_name copa --aug_method knn_15_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name boolq --aug_method knn_15_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name rte --aug_method knn_15_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wsc --aug_method knn_15_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wic --aug_method knn_15_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name multirc --aug_method knn_15_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name cb --aug_method knn_15_0.1 >myout.file 2>&1 &

nohup python -m baseline_aug.total_aug --task_name copa --aug_method synonym_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name boolq --aug_method synonym_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name rte --aug_method synonym_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wsc --aug_method synonym_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name wic --aug_method synonym_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name multirc --aug_method synonym_0.1 >myout.file 2>&1 &
nohup python -m baseline_aug.total_aug --task_name cb --aug_method synonym_0.1 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m baseline_aug.total_aug --task_name copa --aug_method mlm_0.1 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m baseline_aug.total_aug --task_name boolq --aug_method mlm_0.1 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m baseline_aug.total_aug --task_name rte --aug_method mlm_0.1 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 python -m baseline_aug.total_aug --task_name wsc --aug_method mlm_0.1 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m baseline_aug.total_aug --task_name wic --aug_method mlm_0.1 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m baseline_aug.total_aug --task_name multirc --aug_method mlm_0.1 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m baseline_aug.total_aug --task_name cb --aug_method mlm_0.1 >myout.file 2>&1 &


CUDA_VISIBLE_DEVICES=7 python -m baseline_aug.total_aug --task_name copa --aug_method tinybert >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m baseline_aug.total_aug --task_name boolq --aug_method tinybert >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m baseline_aug.total_aug --task_name rte --aug_method tinybert >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m baseline_aug.total_aug --task_name wsc --aug_method tinybert >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m baseline_aug.total_aug --task_name wic --aug_method tinybert >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m baseline_aug.total_aug --task_name multirc --aug_method tinybert >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m baseline_aug.total_aug --task_name cb --aug_method tinybert >myout.file 2>&1 &



CUDA_VISIBLE_DEVICES=7 python -m baseline_aug.total_aug --task_name COPA --aug_method bt_lang_10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m baseline_aug.total_aug --task_name BoolQ --aug_method bt_lang_10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m baseline_aug.total_aug --task_name RTE --aug_method bt_lang_10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m baseline_aug.total_aug --task_name MultiRC --aug_method bt_lang_10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m baseline_aug.total_aug --task_name CB --aug_method bt_lang_10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=7 python -m baseline_aug.total_aug --task_name COPA --aug_method bt_lang_6 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m baseline_aug.total_aug --task_name BoolQ --aug_method bt_lang_6 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m baseline_aug.total_aug --task_name RTE --aug_method bt_lang_6 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m baseline_aug.total_aug --task_name MultiRC --aug_method bt_lang_6 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m baseline_aug.total_aug --task_name CB --aug_method bt_lang_6 >myout.file 2>&1 &


CUDA_VISIBLE_DEVICES=7 python -m baseline_aug.total_aug --task_name copa --aug_method t5_mlm_0.1 >myout_tmlm.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m baseline_aug.total_aug --task_name boolq --aug_method t5_mlm_0.1 >myout_tmlm.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m baseline_aug.total_aug --task_name rte --aug_method t5_mlm_0.1 >myout_tmlm.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m baseline_aug.total_aug --task_name wsc --aug_method t5_mlm_0.1 >myout_tmlm.file 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m baseline_aug.total_aug --task_name wic --aug_method t5_mlm_0.1 >myout_tmlm.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 python -m baseline_aug.total_aug --task_name multirc --aug_method t5_mlm_0.1 >myout_tmlm.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m baseline_aug.total_aug --task_name cb --aug_method t5_mlm_0.1 >myout_tmlm.file 2>&1 &

"""

# CUDA_VISIBLE_DEVICES=4 python -m baseline_aug.total_aug --task_name wic --aug_method t5_mlm_0.1
