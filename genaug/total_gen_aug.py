import os
import json
import copy
import re
import torch
import numpy as np 
import random
import string
from tqdm import tqdm
from genaug.utils import *
from genaug import gen_aug_T5
from nltk.tokenize import sent_tokenize
def line_tokenizer(text):
    return text.split()

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
    def __init__(self,args):
        self.args=args

    def read_jsonl(self,file_path):
        examples=[]
        with open(file_path,"r",encoding="utf8") as f:
            for line in f:
                example_json=json.loads(line)
                examples.append(example_json)
        return examples
        
    def save_jsonl(self,examples,save_path):
        with open(save_path,"w",encoding="utf8") as f:
            for e in examples:
                f.write(json.dumps(e)+'\n')
        f.close()

    def mask_text(self,text,mask_ratio=0.5,cnt=0,substitute_verbalizers=['<extra_id_{}>'.format(i) for i in range(300)],allow_substitute_punctuation=False, at_least_one=False, unchanged_phrases=[],changed_word_list=[]):
        tokens=nltk_line_tokenizer(text)
        n=len(tokens)
        unchanged_phrases=[x.lower() for x in unchanged_phrases]
        splited_unchanged_phrases=[nltk_line_tokenizer(x.lower()) for x in unchanged_phrases]
        changed_word_list=[x.lower() for x in changed_word_list]
        if allow_substitute_punctuation:
            candidate_idxs=np.ones(n)
            for i in range(n):
                for splited_unchanged_phrase in splited_unchanged_phrases:
                    if ' '.join(tokens[i:i+len(splited_unchanged_phrase)]).lower()==' '.join(splited_unchanged_phrase):
                        candidate_idxs[i:i+len(splited_unchanged_phrase)]=0
            candidate_idxs=[i for (i,x) in enumerate(candidate_idxs) if x==1]
            # candidate_idxs=[i for i in range(n) if tokens[i].lower() not in unchanged_word_list]
            idxs_should_be_changed=[i for i in range(n) if tokens[i].lower() in changed_word_list]
            n=len(candidate_idxs)
            indices=sorted(list(set(random.sample(candidate_idxs,int(n*mask_ratio))+idxs_should_be_changed)))
            # indices=sorted(random.sample(range(n),int(n*mask_ratio)))
        else:
            candidate_idxs=np.ones(n)
            for i in range(n):
                for splited_unchanged_phrase in splited_unchanged_phrases:
                    if tokens[i] in string.punctuation:
                        candidate_idxs[i]=0
                    if ' '.join(tokens[i:i+len(splited_unchanged_phrase)]).lower()==' '.join(splited_unchanged_phrase):
                        candidate_idxs[i:i+len(splited_unchanged_phrase)]=0
            candidate_idxs=[i for (i,x) in enumerate(candidate_idxs) if x==1]
            # candidate_idxs=[i for i in range(n) if tokens[i] not in string.punctuation and tokens[i].lower() not in unchanged_word_list]
            idxs_should_be_changed=[i for i in range(n) if tokens[i].lower() in changed_word_list]
            n=len(candidate_idxs)
            indices=sorted(list(set(random.sample(candidate_idxs,int(n*mask_ratio))+idxs_should_be_changed)))
        if at_least_one==True and len(indices)==0:
            indices=sorted(random.sample(range(n),1))
        masked_src, masked_tgt = "", []
        for i, idx in enumerate(indices):
            if i == 0 or idx != indices[i-1] + 1:
                masked_tgt.append("")
            masked_tgt[-1] += " " + tokens[idx]
            tokens[idx] = "[MASK]"
        for i, token in enumerate(tokens):
            if i != 0 and token == "[MASK]" and tokens[i-1] == "[MASK]":
                continue
            if token=="[MASK]":
                masked_src+=" "+substitute_verbalizers[cnt]
                cnt+=1
            else:
                masked_src += " " + token
        return masked_src.strip(), masked_tgt, cnt

    def predict_blanks(self,texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type='default'):
        print('def predict_blanks.aug_kwargs:{},aug_type:{}'.format(aug_kwargs,aug_type))
        if 'iter' in aug_type:
            batch_size=int(aug_type.split('_')[2])
            pred_blanks=[]
            for (text_to_be_augmented,tgt_parts) in zip(texts_to_be_augmented,tgt_texts):
                blen=len(tgt_parts)
                new_tgt_parts=copy.deepcopy(tgt_parts)
                masked_idxs=list(range(blen))
                if aug_type.startswith('rand_iter'):
                    random.shuffle(masked_idxs)
                text_parts=re.split('<extra_id_\d+>',text_to_be_augmented)
                for batch_idx in range(int(np.ceil(len(masked_idxs)/batch_size))):
                    cnt=0
                    masked_id=masked_idxs[batch_idx*batch_size:(batch_idx+1)*batch_size]
                    masked_id=sorted(masked_id)
                    new_text=''
                    for i in range(len(text_parts)-1):
                        new_text+=text_parts[i]
                        if i in masked_id:
                            new_text+='<extra_id_{}>'.format(cnt)
                            cnt+=1
                        else:
                            new_text+=new_tgt_parts[i]
                    new_text+=text_parts[-1]
                    total_predictions,preds=gen_blanks_func([new_text],**aug_kwargs)
                    preds=preds[0][0]
                    # print(new_text,preds)
                    if len(preds)>len(masked_id):
                        preds=preds[:len(masked_id)]
                    else:
                        for _ in range(len(masked_id)-len(preds)):
                            preds.append('')
                    for (m_id,pred_blank) in zip(masked_id,preds):
                        new_tgt_parts[m_id]=pred_blank
                pred_blanks.append(new_tgt_parts)
        elif aug_type=='default':
            total_predictions, pred_blanks=gen_blanks_func(texts_to_be_augmented,**aug_kwargs)
            pred_blanks=[pred_blank[0] for pred_blank in pred_blanks]
            # pred_blanks=pred_blanks[0]
        return pred_blanks
        
    def recover_examples_from_blanks(self,pure_parts,pred_blanks,model_type='t5'):
        # example_lines=[['[MASK] x','[MASK] y'],['x [MASK] y', '[MASK] z']]
        # pred_blanks=[['a','b'],['c','d']]
        # return filled_parts=[['a x','b y'],['x c y','d z']]
        if model_type is None:
            lines=' '.join([' '.join(x) for x in pure_parts])
            if '[MASK]' in lines:
                model_type='GLM'
            elif '<extra_id_0>' in lines:
                model_type='t5'
        filled_parts=[]
        for (parts,pred_blank) in zip(pure_parts,pred_blanks):
            current_blank=0
            filled_parts.append([])
            for part in parts:
                output_tokens=[]
                tokens=part.split()
                for token in tokens:
                    if (model_type.lower()=='t5' and token.startswith('<extra_id_')) or (model_type.lower=='glm' and token.startswith('[MASK]')):
                        if current_blank < len(pred_blank):
                            output_tokens.append(pred_blank[current_blank])
                        current_blank+=1
                    else:
                        output_tokens.append(token)
                filled_parts[-1].append(' '.join((' '.join(output_tokens)).split()).strip())
                # print('def recover_examples_from_blanks',filled_parts[-1])
        return filled_parts

    def postprocess_texts(self,filled_parts):
        processed_parts=[]
        for parts in filled_parts:
            processed_parts.append([])
            for part in parts:
                processed_parts[-1].append(part.strip(string.punctuation).strip())
        return processed_parts

class RTEAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="RTE"

    def aug_with_pattern(self,question,passage,label,gen_blanks_func,aug_kwargs,label_type='flip',mask_ratio=0.5,aug_type='rand_iter',aug_num=1):
        bad_words_ids=[[3], [19794], [22354]]+[[2163],[4273],[465],[150],[1525],[58]] # " ","........","................","Yes","yes","No","no"," answer","?"
        aug_kwargs['bad_words_ids']=bad_words_ids
        texts_to_be_augmented=[];tgt_texts=[]
        masked_parts=[]
        new_questions=[];new_passages=[];new_labels=[]
        for aug_idx in range(aug_num):
            if label_type=='flip':
                if label=='entailment': label_text='No'; new_label='not_entailment'
                elif label=='not_entailment': label_text="Yes";new_label="entailment"
            else:
                if label=='entailment': label_text='Yes'
                elif label=='not_entailment': label_text="No"
                new_label=label
            new_labels.append(new_label)
            masked_question,question_tgt,question_cnt=self.mask_text(question,mask_ratio=mask_ratio)
            masked_passage,passage_tgt,passage_cnt=self.mask_text(passage,cnt=question_cnt,mask_ratio=mask_ratio)
            texts_to_be_augmented.append(masked_question+'?'+label_text+', '+masked_passage)
            masked_parts.append([masked_question,masked_passage])
            # texts_to_be_augmented=[masked_passage+'. Based on the previous passage, '+masked_question+'?'+label_text]
            tgt_texts.append(question_tgt+passage_tgt)
        pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
        filled_parts=self.recover_examples_from_blanks(masked_parts,pred_blanks)
        filled_parts=self.postprocess_texts(filled_parts)
        for parts in filled_parts:
            [new_question,new_passage]=parts
            new_questions.append(new_question)
            new_passages.append(new_passage)
        return new_questions,new_passages,new_labels

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in tqdm(examples):
            new_premise,new_hypothesis,new_label=self.aug_with_pattern(e["hypothesis"],e["premise"],e["label"],aug_func,**aug_kwargs,aug_num=aug_num)
            if isinstance(new_premise,list):
                for (x,y,z) in zip(new_premise,new_hypothesis,new_label):
                    tmp_e=copy.deepcopy(e)
                    tmp_e["hypothesis"]=x 
                    tmp_e["premise"]=y
                    tmp_e["label"]=z
                    tmp_e["orig_label"]=e["label"]
                    new_examples.append(tmp_e)
            else:
                tmp_e=copy.deepcopy(e)
                tmp_e["hypothesis"]=new_premise
                tmp_e["premise"]=new_hypothesis
                tmp_e["label"]=new_label
                tmp_e["orig_label"]=e["label"]
                new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

class CBAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="CB"

    def aug_with_pattern(self,question,passage,label,gen_blanks_func,aug_kwargs,label_type='flip',mask_ratio=0.5,aug_type='rand_iter',aug_num=1):
        bad_words_ids=[[3], [19794], [22354]]+[[2163],[4273],[465],[150],[1525],[58]] 
        aug_kwargs['bad_words_ids']=bad_words_ids
        texts_to_be_augmented=[];tgt_texts=[]
        masked_parts=[]
        new_questions=[];new_passages=[];new_labels=[]
        for aug_idx in range(aug_num):
            if label_type=='flip':
                if label=='entailment': 
                    if random.random()<0.5: label_text='No'; new_label="contradiction"
                    else: label_text="Maybe";new_label="neutral"
                elif label=='contradiction': 
                    if random.random()<0.5: label_text="Yes";new_label="entailment"
                    else: label_text="Maybe";new_label="neutral"
                elif label=='neutral': 
                    if random.random()<0.5: label_text="Yes";new_label="entailment"
                    else: label_text="No";new_label="contradiction"
            else:
                if label=='entailment': label_text='Yes'
                elif label=='contradiction': label_text="No"
                elif label=='neutral': label_text="Maybe"
                new_label=label
            new_labels.append(new_label)
            masked_question,question_tgt,question_cnt=self.mask_text(question,mask_ratio=mask_ratio)
            masked_passage,passage_tgt,passage_cnt=self.mask_text(passage,cnt=question_cnt,mask_ratio=mask_ratio)
            texts_to_be_augmented.append('"'+masked_question+'" ?'+label_text+'. "'+masked_passage+'"')
            masked_parts.append([masked_question,masked_passage])
            tgt_texts.append(question_tgt+passage_tgt)
        pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
        filled_parts=self.recover_examples_from_blanks(masked_parts,pred_blanks)
        filled_parts=self.postprocess_texts(filled_parts)
        for parts in filled_parts:
            [new_question,new_passage]=parts
            new_questions.append(new_question)
            new_passages.append(new_passage)
        return new_questions,new_passages,new_labels

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in tqdm(examples):
            new_premise,new_hypothesis,new_label=self.aug_with_pattern(e["hypothesis"],e["premise"],e["label"],aug_func,**aug_kwargs,aug_num=aug_num)
            if isinstance(new_premise,list):
                for (x,y,z) in zip(new_premise,new_hypothesis,new_label):
                    tmp_e=copy.deepcopy(e)
                    tmp_e["hypothesis"]=x 
                    tmp_e["premise"]=y
                    tmp_e["label"]=z
                    tmp_e["orig_label"]=e["label"]
                    new_examples.append(tmp_e)
            else:
                tmp_e=copy.deepcopy(e)
                tmp_e["hypothesis"]=new_premise
                tmp_e["premise"]=new_hypothesis
                tmp_e["label"]=new_label
                tmp_e["orig_label"]=e["label"]
                new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

class BoolQAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="BoolQ"

    def aug_with_pattern(self,question,passage,label,gen_blanks_func,aug_kwargs,label_type='flip',mask_ratio=0.5,aug_type='rand_iter',aug_num=1):
        bad_words_ids=[[3], [19794], [22354]]+[[2163],[4273],[465],[150],[1525],[58]] 
        aug_kwargs['bad_words_ids']=bad_words_ids
        texts_to_be_augmented=[];tgt_texts=[]
        masked_parts=[]
        new_questions=[];new_passages=[];new_labels=[]
        for aug_idx in range(aug_num):
            if label_type=='flip': new_label=bool(1-label)
            else: new_label=bool(label)
            if new_label==True: label_text="Yes"
            else: label_text="No"
            new_labels.append(new_label)
            masked_question,question_tgt,question_cnt=self.mask_text(question,mask_ratio=mask_ratio)
            masked_passage,passage_tgt,passage_cnt=self.mask_text(passage,cnt=question_cnt,mask_ratio=mask_ratio)
            texts_to_be_augmented.append(masked_question+'?'+label_text+', '+masked_passage)
            tgt_texts.append(question_tgt+passage_tgt)
            masked_parts.append([masked_question,masked_passage])
        pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
        filled_parts=self.recover_examples_from_blanks(masked_parts,pred_blanks)
        filled_parts=self.postprocess_texts(filled_parts)
        # print(texts_to_be_augmented[0],'\n',masked_parts[0],'\n',tgt_texts[0],'\n',filled_parts[0])
        for parts in filled_parts:
            [new_question,new_passage]=parts
            new_questions.append(new_question)
            new_passages.append(new_passage)
        return new_questions,new_passages,new_labels

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in tqdm(examples):
            new_premise,new_hypothesis,new_label=self.aug_with_pattern(e["question"],e["passage"],e["label"],aug_func,**aug_kwargs,aug_num=aug_num)
            if isinstance(new_premise,list):
                for (x,y,z) in zip(new_premise,new_hypothesis,new_label):
                    tmp_e=copy.deepcopy(e)
                    tmp_e["question"]=x 
                    tmp_e["passage"]=y
                    tmp_e["label"]=z
                    tmp_e["orig_label"]=e["label"]
                    new_examples.append(tmp_e)
            else:
                tmp_e=copy.deepcopy(e)
                tmp_e["question"]=new_premise
                tmp_e["passage"]=new_hypothesis
                tmp_e["label"]=new_label
                tmp_e["orig_label"]=e["label"]
                new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples


class COPAAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="COPA"

    def aug_with_pattern(self,premise,choice1,choice2,question,label,gen_blanks_func,aug_kwargs,label_type='flip',mask_ratio=0.5,aug_type='rand_iter',aug_num=1,global_premises=None):
        bad_words_ids=[[3], [19794], [22354]]+[[5],[58],[6]]
        aug_kwargs['bad_words_ids']=bad_words_ids
        premise=premise.strip(string.punctuation)
        choices=[choice1,choice2]
        texts_to_be_augmented=[];tgt_texts=[];masked_parts=[]
        texts_to_be_augmented2=[];tgt_texts2=[];masked_parts2=[]
        new_premises=[];new_choice1s=[];new_choice2s=[];new_labels=[];new_questions=[]
        for aug_idx in range(aug_num):
            if label_type=='flip':
                new_label=1-label
                if random.random()<0.5:
                    new_question='effect' if question=='cause' else 'cause'
                else: new_question=question
            else:
                new_label=label
                new_question=question
            new_labels.append(new_label)
            new_questions.append(new_question)
            if new_question=='effect': label_text=', so that '
            else: label_text=', because '
            masked_premise,premise_tgt,premise_cnt=self.mask_text(premise,cnt=0,mask_ratio=mask_ratio)
            texts_to_be_augmented.append(masked_premise+label_text+choices[new_label])
            tgt_texts.append(premise_tgt)
            masked_parts.append([masked_premise])
        pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
        filled_parts=self.recover_examples_from_blanks(masked_parts,pred_blanks)
        filled_parts=self.postprocess_texts(filled_parts)
        for (i,parts) in enumerate(filled_parts):
            [new_premise]=parts
            new_premises.append(new_premise+'.')
            new_choice1s.append(choice1)
            new_choice2s.append(choice2)
        return new_premises,new_choice1s,new_choice2s,new_questions,new_labels

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        global_premises=[e['premise'] for e in examples]
        for e in examples:
            new_premise,new_choice1,new_choice2,new_question,new_label=self.aug_with_pattern(e["premise"],e["choice1"],e["choice2"],e["question"],e["label"],aug_func,**aug_kwargs,aug_num=aug_num,global_premises=global_premises)
            if isinstance(new_premise,list):
                for (x,y,z,nq,nl) in zip(new_premise,new_choice1,new_choice2,new_question,new_label):
                    tmp_e=copy.deepcopy(e)
                    tmp_e["premise"]=x 
                    tmp_e["choice1"]=y 
                    tmp_e["choice2"]=z
                    tmp_e["question"]=nq
                    tmp_e["label"]=nl
                    tmp_e["orig_label"]=e["label"]
                    new_examples.append(tmp_e)
            else:
                tmp_e=copy.deepcopy(e)
                tmp_e["premise"]=new_premise
                tmp_e["choice1"]=new_choice1
                tmp_e["choice2"]=new_choice2
                tmp_e["question"]=new_question
                tmp_e["label"]=new_label
                tmp_e["orig_label"]=e["label"]
                new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

class WiCAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="WiC"

    def aug_with_pattern(self,texts,word,word1,word2,label,gen_blanks_func,aug_kwargs,label_type='flip',mask_ratio=0.5,aug_type='rand_iter',aug_num=1,global_premises=None):
        bad_words_ids=[[3], [19794], [22354]]+[[5],[58],[6]]+[[337],[315]] # ".?, same different"
        aug_kwargs['bad_words_ids']=bad_words_ids
        if label_type=='flip': new_label=bool(1-label)
        else: new_label=bool(label)
        if new_label==False:
            texts_to_be_augmented1=[];tgt_texts1=[];masked_parts1=[]
            texts_to_be_augmented2=[];tgt_texts2=[];masked_parts2=[]
            new_textss=[];new_labels=[]
            for aug_idx in range(aug_num):
                if new_label==True: label_text="the same"
                else: label_text="different"
                new_labels.append(new_label)
                masked1,tgt1,cnt1=self.mask_text(texts[0],mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                masked2,tgt2,cnt2=self.mask_text(texts[1],cnt=cnt1,mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                masked3,tgt3,cnt3=self.mask_text(texts[2],cnt=0,mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                masked4,tgt4,cnt4=self.mask_text(texts[3],cnt=cnt3,mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                texts_to_be_augmented1.append(np.random.choice(global_premises)+' '+masked1+' '+word1+' '+masked2)
                texts_to_be_augmented2.append(np.random.choice(global_premises)+' '+masked3+' '+word2+' '+masked4)
                tgt_texts1.append(tgt1+tgt2)
                tgt_texts2.append(tgt3+tgt4)
                masked_parts1.append([masked1,masked2])
                masked_parts2.append([masked3,masked4])
            aug_kwargs['min_length']=2+int(mask_ratio*(len(texts[0].split())+len(texts[1].split())))
            pred_blanks1=self.predict_blanks(texts_to_be_augmented1,tgt_texts1,gen_blanks_func,aug_kwargs,aug_type=aug_type)
            filled_parts1=self.recover_examples_from_blanks(masked_parts1,pred_blanks1)
            new_textss1=self.postprocess_texts(filled_parts1)
            aug_kwargs['min_length']=2+int(mask_ratio*(len(texts[2].split())+len(texts[3].split())))
            pred_blanks2=self.predict_blanks(texts_to_be_augmented2,tgt_texts2,gen_blanks_func,aug_kwargs,aug_type=aug_type)
            filled_parts2=self.recover_examples_from_blanks(masked_parts2,pred_blanks2)
            new_textss2=self.postprocess_texts(filled_parts2)
            new_textss=[]
            for x,y in zip(new_textss1,new_textss2):
                new_textss.append([])
                new_textss[-1]=x+y
        elif new_label==True:
            texts_to_be_augmented=[];tgt_texts=[];masked_parts=[]
            new_textss=[];new_labels=[]
            for aug_idx in range(aug_num):
                if new_label==True: label_text="the same"
                else: label_text="different"
                new_labels.append(new_label)
                masked1,tgt1,cnt1=self.mask_text(texts[0],mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                masked2,tgt2,cnt2=self.mask_text(texts[1],cnt=cnt1,mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                masked3,tgt3,cnt3=self.mask_text(texts[2],cnt=cnt2,mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                masked4,tgt4,cnt4=self.mask_text(texts[3],cnt=cnt3,mask_ratio=mask_ratio,allow_substitute_punctuation=True)
                texts_to_be_augmented.append(masked1+' '+word1+' '+masked2+' . '+masked3+' '+word2+' '+masked4 + 'Word "'+word+'"' + ' means {} in the two sentences'.format(label_text))
                tgt_texts.append(tgt1+tgt2+tgt3+tgt4)
                masked_parts.append([masked1,masked2,masked3,masked4])
            pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
            filled_parts=self.recover_examples_from_blanks(masked_parts,pred_blanks)
            new_textss=self.postprocess_texts(filled_parts)
        return new_textss,new_labels

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        global_premises=[e['sentence1'] for e in examples]+[e['sentence2'] for e in examples]
        for e in examples:
            seq1=nltk_line_tokenizer(e["sentence1"])
            seq2=nltk_line_tokenizer(e["sentence2"])
            word=e["word"]
            idx1,idx2=seq1.index(e["sentence1"][e["start1"]:e["end1"]]),seq2.index(e["sentence2"][e["start2"]:e["end2"]])
            sentences_to_be_augmented=[]
            sentences_to_be_augmented.append(" ".join(seq1[:idx1]))
            sentences_to_be_augmented.append(" ".join(seq1[idx1+1:]))
            sentences_to_be_augmented.append(" ".join(seq2[:idx2]))
            sentences_to_be_augmented.append(" ".join(seq2[idx2+1:]))
            total_sentences,new_labels=self.aug_with_pattern(sentences_to_be_augmented,e["word"],e["sentence1"][e["start1"]:e["end1"]],e["sentence2"][e["start2"]:e["end2"]],e["label"],aug_func,**aug_kwargs,aug_num=aug_num,global_premises=global_premises)
            if isinstance(total_sentences,list):
                for ((a,b,c,d),f) in zip(total_sentences,new_labels):
                    tmp_e=copy.deepcopy(e)
                    tmp_e["sentence1"]=" ".join(a.split()+[e["sentence1"][e["start1"]:e["end1"]]]+b.split())
                    tmp_e["sentence2"]=" ".join(c.split()+[e["sentence2"][e["start2"]:e["end2"]]]+d.split())
                    tmp_e["start1"]=len(a)+1
                    tmp_e["start2"]=len(c)+1
                    tmp_e["end1"]=len(a)+e["end1"]-e["start1"]+1
                    tmp_e["end2"]=len(c)+e["end2"]-e["start2"]+1
                    tmp_e["label"]=f
                    tmp_e["orig_label"]=e["label"]
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
                tmp_e["label"]=new_label
                tmp_e["orig_label"]=e["label"]
                new_examples.append(tmp_e)

        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

'''
from genaug import total_gen_aug
myaug=total_gen_aug.WSCAug()
filtered_nouns=myaug.prepare_wsc_nouns()
'''
class WSCAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="WSC"

    def aug_with_pattern(self,texts,word1,word2,gen_blanks_func,aug_kwargs,label_type='keep',mask_ratio=0.5,aug_type='rand_iter'):
        if label_type=='flip':
            raise NotImplementedError
        bad_words_ids=[[3], [19794], [22354]]
        aug_kwargs['bad_words_ids']=bad_words_ids
        masked1,tgt1,cnt1=self.mask_text(texts[0],mask_ratio=mask_ratio)
        masked2,tgt2,cnt2=self.mask_text(texts[1],cnt=cnt1,mask_ratio=mask_ratio)
        masked3,tgt3,cnt3=self.mask_text(texts[2],cnt=cnt2,mask_ratio=mask_ratio)
        texts_to_be_augmented=[masked1.strip()+' '+word1+' '+masked2.strip()+' '+word2+' '+masked3.strip()]
        # print(texts_to_be_augmented)
        # texts_to_be_augmented=[masked_passage+'. Based on the previous passage, '+masked_question+'?'+label_text]
        tgt_texts=[tgt1+tgt2+tgt3]
        pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
        filled_parts=self.recover_examples_from_blanks([[masked1,masked2,masked3]],pred_blanks)[0]
        new_texts=[x.strip(string.punctuation).strip() for x in filled_parts]
        return new_texts

    def aug_wsc_extra(self,e,aug_func,aug_func_name,aug_kwargs):
        seq=line_tokenizer(e["text"])
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
        total_sentences=self.aug_with_pattern(sentences_to_be_augmented,' '.join(fill_text1),' '.join(fill_text2),aug_func,**aug_kwargs)
        if isinstance(total_sentences[0],list):
            print('not implemented')
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
        return tmp_e

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1,wsc_aug_type='np_extra'):
        '''
        extra: only substitute other words except e['word']
        '''
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in examples:
            for _ in range(aug_num):
                if wsc_aug_type=='extra':
                    tmp_e=self.aug_wsc_extra(e,aug_func,aug_func_name,aug_kwargs)
                else:
                    raise NotImplementedError
                new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples


class MultiRCAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="MultiRC"

    def aug_with_pattern(self,question,passage,answer,label,gen_blanks_func,aug_kwargs,label_type='flip',mask_ratio=0.5,aug_type='rand_iter',aug_num=1):
        bad_words_ids=[[3], [19794], [22354]]+[[5],[58],[6]] # ".,?"
        aug_kwargs['bad_words_ids']=bad_words_ids
        texts_to_be_augmented=[];tgt_texts=[];masked_parts=[]
        new_questions=[];new_passages=[];new_answers=[];new_labels=[]
        for aug_idx in range(aug_num):
            if label_type=='flip': new_label=(1-label)
            else: new_label=(label)
            if new_label==True: label_text="Yes"
            else: label_text="No"
            masked_question,question_tgt,question_cnt=self.mask_text(question,mask_ratio=mask_ratio)
            masked_answer,answer_tgt,answer_cnt=self.mask_text(answer,cnt=question_cnt,mask_ratio=mask_ratio)
            masked_passage,passage_tgt,passage_cnt=self.mask_text(passage,cnt=answer_cnt,mask_ratio=mask_ratio)
            texts_to_be_augmented.append(masked_question + '? Is the correct answer "'+masked_answer+'"?'+label_text+'. '+masked_passage)
            tgt_texts.append(question_tgt+answer_tgt+passage_tgt)
            masked_parts.append([masked_question,masked_answer,masked_passage])
            new_labels.append(new_label)
        pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
        filled_parts=self.recover_examples_from_blanks(masked_parts,pred_blanks)
        filled_parts=self.postprocess_texts(filled_parts)
        for parts in filled_parts:
            [new_question,new_answer,new_passage]=parts
            new_questions.append(new_question)
            new_passages.append(new_passage)
            new_answers.append(new_answer)
        return new_questions,new_passages,new_answers,new_labels

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl").format(self.TASK_NAME))
        new_examples=[]
        for e in examples:
            for (i,answer) in enumerate(e['passage']['questions'][0]['answers']):
                new_questions,new_passages,new_answers,new_labels=self.aug_with_pattern(e["passage"]["questions"][0]["question"],\
                    e["passage"]["text"],answer['text'],answer['label'],aug_func,**aug_kwargs,aug_num=aug_num)
                if isinstance(new_questions,list):
                    for new_question,new_passage,new_answer,new_label in zip(new_questions,new_passages,new_answers,new_labels):
                        tmp_e=copy.deepcopy(e)
                        tmp_e["passage"]["text"]=new_passage
                        tmp_e["passage"]["questions"][0]["question"]=new_question
                        tmp_e["passage"]["questions"][0]["answers"]=[{"text": new_answer, "idx": answer['idx'], "label":new_label,"orig_label":answer['label']}]
                        new_examples.append(tmp_e)
                else:
                    new_question,new_passage,new_answer,new_label=new_questions,new_passages,new_answers,new_labels
                    tmp_e=copy.deepcopy(e)
                    tmp_e["passage"]["text"]=new_passage
                    tmp_e["passage"]["questions"][0]["question"]=new_question
                    tmp_e["passage"]["questions"][0]["answers"]=[{"text": new_answer, "idx": answer['idx'], "label":new_label,"orig_label":answer['label']}]
                    new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples


class ReCoRDAug(FewGLUEAug):
    def __init__(self,args):
        super().__init__(args)
        self.TASK_NAME="ReCoRD"
        from transformers import T5Tokenizer
        self.tokenizer=T5Tokenizer.from_pretrained(self.args.model_name_or_path)

    def aug_with_pattern(self,passage,question,entities,answers,gen_blanks_func,aug_kwargs,label_type='flip',mask_ratio=0.5,aug_type='rand_iter',aug_num=1):
        bad_words_ids=[[3], [19794], [22354]]
        aug_kwargs['bad_words_ids']=bad_words_ids
        texts_to_be_augmented=[];tgt_texts=[]
        masked_parts=[]
        new_questions=[];new_passages=[];new_entities=[];new_answers=[]
        aug_idx=0
        patience=10
        retry_number=0
        while(aug_idx<aug_num):
            if label_type=='flip':
                new_answer=np.random.choice([x for x in entities if x not in answers and x not in question])
                new_answers.append([new_answer])
            else:
                new_answers.append(answers)
            new_entities.append(entities)
            masked_question,question_tgt,question_cnt=self.mask_text(question.replace('@placeholder',new_answers[-1][0]),mask_ratio=mask_ratio,unchanged_phrases=new_answers[-1])
            if label_type=='flip':
                masked_passage,passage_tgt,passage_cnt=self.mask_text(passage,cnt=question_cnt,mask_ratio=mask_ratio,changed_word_list=list(set([y for x in answers for y in x.split()])))
            else:
                masked_passage,passage_tgt,passage_cnt=self.mask_text(passage,cnt=question_cnt,mask_ratio=mask_ratio)
            new_answer=' '.join(nltk_line_tokenizer(new_answers[aug_idx][0]))
            replaced_mask_question=masked_question.replace(new_answer,'@placeholder').replace(new_answers[aug_idx][0],'@placeholder')
            if replaced_mask_question.count('@placeholder')!=1:
                print('retry')
                retry_number+=1
                if retry_number>patience:
                    start_pos=masked_question.find(new_answer)
                    end_pos=start_pos+len(new_answer)
                    replaced_mask_question=masked_question[:start_pos]+'@placeholder'+masked_question[end_pos:]
                else:
                    new_answers=new_answers[:-1]
                    new_entities=new_entities[:-1]
                    continue
            texts_to_be_augmented.append(masked_question+masked_passage)
            tgt_texts.append(question_tgt+passage_tgt)
            masked_parts.append([replaced_mask_question,masked_passage])
            aug_idx+=1
        pred_blanks=self.predict_blanks(texts_to_be_augmented,tgt_texts,gen_blanks_func,aug_kwargs,aug_type=aug_type)
        filled_parts=self.recover_examples_from_blanks(masked_parts,pred_blanks)
        for (aug_idx,parts) in enumerate(filled_parts):
            [new_question,new_passage]=parts
            new_passages.append(new_passage)
            if new_question.count('@placeholder')!=1:
                raise SyntaxError('could only have one @placeholder in the question')
            new_questions.append(new_question)
        return new_passages,new_questions,new_entities,new_answers

    def get_entities_and_ans(self,e):
        entities = set()
        for entity_json in e['passage']['entities']:
            start = entity_json['start']
            end = entity_json['end']
            entity = e['passage']['text'][start:end + 1]
            entities.add(entity)
        entities = list(entities)
        answers=set()
        for answer_json in e['qas'][0].get('answers', []):
            answer = answer_json['text']
            answers.add(answer)
        answers = list(answers)
        return entities, answers

    def augment(self,data_path,aug_func,aug_func_name,aug_kwargs,aug_num=1):
        examples=self.read_jsonl(os.path.join(data_path,"{}/train.jsonl".format(self.TASK_NAME)))
        new_examples=[]
        for e in tqdm(examples):
            entities, answers=self.get_entities_and_ans(e)
            new_passages,new_questions,new_entities,new_answers=self.aug_with_pattern(e["passage"]["text"].replace("@highlight\n", "- "),e["qas"][0]["query"],entities,answers,aug_func,**aug_kwargs,aug_num=aug_num)
            if isinstance(new_passages,list):
                for (x,y,z,w) in zip(new_passages,new_questions,new_entities,new_answers):
                    tmp_e=copy.deepcopy(e)
                    tmp_e['passage']['entities']=[]
                    tmp_e["passage"]["text"]=x 
                    tmp_e["qas"][0]["query"]=y
                    tmp_e["passage"]["entity_names"]=z
                    tmp_e["qas"][0]["answers"]=[{'start': -1, 'end': -1, 'text': ans} for ans in w]
                    new_examples.append(tmp_e)
            else:
                tmp_e=copy.deepcopy(e)
                tmp_e['passage']['entities']=[]
                tmp_e["passage"]["text"]=new_passages
                tmp_e["qas"][0]["query"]=new_questions
                tmp_e["passage"]["entity_names"]=new_entities
                tmp_e["qas"][0]["answers"]=[{'start': -1, 'end': -1, 'text': ans} for ans in new_answers]
                new_examples.append(tmp_e)
        if not os.path.exists(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME))):
            os.makedirs(os.path.join(data_path,"augmented/{}".format(self.TASK_NAME)))
        self.save_jsonl(new_examples,os.path.join(data_path,"augmented/{}/{}_train.jsonl".format(self.TASK_NAME,aug_func_name)))
        return new_examples

FEWGLUEAUGS = {
    "WSC": WSCAug,
    "RTE": RTEAug,
    "CB": CBAug,
    "WiC": WiCAug,
    "BoolQ": BoolQAug,
    "COPA": COPAAug,
    "MultiRC": MultiRCAug,
    "ReCoRD":ReCoRDAug
} 

import argparse
def init_args():
    parser = argparse.ArgumentParser(description="Command line interface for PET/iPET")
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--mask_ratio",required=True,type=float)
    parser.add_argument("--label_type",type=str,default="flip")
    parser.add_argument("--aug_type",type=str,default='rand_iter_10')
    parser.add_argument("--do_sample",action="store_true")
    parser.add_argument("--num_beams",type=int,default=1)
    parser.add_argument("--aug_num",type=int,default=10)
    parser.add_argument("--wsc_aug_type",type=str,default='np_extra')
    parser.add_argument("--model_name_or_path",type=str,default='t5-large')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args=init_args()
    myaug=FEWGLUEAUGS[args.task_name](args)
    data_path="data/FewGLUE_dev32/"
    if os.path.exists(data_path)==False:
        os.makedirs(data_path)
    set_seed(1)
    aug_num=args.aug_num
    t5aug=gen_aug_T5.T5Aug(model_path=args.model_name_or_path)
    aug_func=t5aug.generate_blanks
    aug_func_name='t5_{}_{}_{}_sample{}_beam{}_augnum{}'.format(args.label_type,args.mask_ratio,args.aug_type,int(args.do_sample),args.num_beams,aug_num)
    aug_kwargs={'label_type':args.label_type,'mask_ratio':args.mask_ratio,'aug_type':args.aug_type,'aug_kwargs':{'do_sample':args.do_sample,'num_beams':args.num_beams,'num_return_sequences':1}}
    print(aug_kwargs)
    if args.task_name.lower()=='wsc':
        aug_func_name+='wscaugtype_{}'.format(args.wsc_aug_type)
        new_examples=myaug.augment(data_path,aug_func,aug_func_name,aug_kwargs,aug_num=aug_num,wsc_aug_type=args.wsc_aug_type)
    else:
        new_examples=myaug.augment(data_path,aug_func,aug_func_name,aug_kwargs,aug_num=aug_num)


'''
CUDA_VISIBLE_DEVICES=6 python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'default' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10
CUDA_VISIBLE_DEVICES=7 python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'rand_iter_10' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10


CUDA_VISIBLE_DEVICES=0 python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'rand_iter_10' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'rand_iter_10' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'rand_iter_10' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'default' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'default' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'default' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

# todo 
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'rand_iter_10' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'rand_iter_10' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'rand_iter_10' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'default' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'default' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.1 --aug_type 'default' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'rand_iter_10' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'rand_iter_10' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'rand_iter_10' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'default' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'default' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'default' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &


CUDA_VISIBLE_DEVICES=5 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'rand_iter_10' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'rand_iter_10' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'rand_iter_10' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'default' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'default' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.3 --aug_type 'default' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'rand_iter_10' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'rand_iter_10' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'rand_iter_10' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'rand_iter_10' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'rand_iter_10' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'rand_iter_10' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name ReCoRD --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

'''


'''
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.1 --aug_type 'rand_iter_1' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.1 --aug_type 'rand_iter_1' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.1 --aug_type 'rand_iter_1' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=4 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.1 --aug_type 'default' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.1 --aug_type 'default' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.1 --aug_type 'default' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'rand_iter_1' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'rand_iter_1' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'rand_iter_1' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'default' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'default' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'default' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'rand_iter_1' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'rand_iter_1' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'rand_iter_1' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'default' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'default' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.3 --aug_type 'default' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=4 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'rand_iter_1' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'rand_iter_1' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'rand_iter_1' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'rand_iter_1' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'rand_iter_1' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'rand_iter_1' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=5 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'rand_iter_1' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'rand_iter_1' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'rand_iter_1' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'default' --label_type 'keep' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'default' --label_type 'keep' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=6 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'default' --label_type 'keep' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'rand_iter_1' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'rand_iter_1' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'rand_iter_1' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'default' --label_type 'flip' --do_sample --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=3 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'default' --label_type 'flip' --aug_num 10 >myout.file 2>&1 &
CUDA_VISIBLE_DEVICES=7 nohup python -m genaug.total_gen_aug --task_name WSC --mask_ratio 0.8 --aug_type 'default' --label_type 'flip' --num_beams 10 --aug_num 10 >myout.file 2>&1 &

'''
