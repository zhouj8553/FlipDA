from tqdm import trange,tqdm
import torch
import transformers

from transformers import AlbertForMaskedLM
from pet.wrapper import TransformerModelWrapper, SEQUENCE_CLASSIFIER_WRAPPER, WrapperConfig
import itertools
import numpy as np
import copy
import string
import re

from scipy.special import softmax
from collections import defaultdict
'''
path='/workspace/zhoujing/FewGLUE_dev32/augmented/RTE/t5_flip_label_0.5_default_augnum10_train.jsonl'
from pet.tasks import RteProcessor
myprocessor=RteProcessor()
new_examples=myprocessor._create_examples(path,'aug')
from genaug import confidence_filter
import imp
imp.reload(confidence_filter)
# myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='/workspace/zhoujing/pet_master/results/baseline/pet/rte_albert_model/p2-i0')
myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='new_results/baseline/pet/rte_albert_model/p2-i0')
import pet
eval_config = pet.EvalConfig(device='cuda:0', n_gpu=1, metrics='acc', per_gpu_eval_batch_size=1, decoding_strategy='default', priming=False)
# outputs=myfilter.validate(myfilter.wrapper,eval_data,eval_config)
filtered_examples=myfilter.recover_labels(myfilter.wrapper,new_examples,eval_config,recover_type='deterministic_topk')

import imp
imp.reload(confidence_filter)
newfilter=confidence_filter.Confidence_Filter()
filtered_examples=newfilter.recover_labels(myfilter.wrapper,new_examples,eval_config,recover_type='deterministic_topk')


path='/workspace/zhoujing/FewGLUE_dev32/augmented/CB/t5_flip_0.5_rand_iter_10_sample1_beam1_augnum10_train.jsonl'
from pet.tasks import CbProcessor
myprocessor=CbProcessor()
new_examples=myprocessor._create_examples(path,'aug')
from genaug import confidence_filter
import imp
imp.reload(confidence_filter)
myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='/workspace/zhoujing/pet_master/results/baseline/pet/cb_albert_model/p2-i0')
import pet
eval_config = pet.EvalConfig(device='cuda:0', n_gpu=1, metrics='acc', per_gpu_eval_batch_size=1, decoding_strategy='default', priming=False)
# outputs=myfilter.validate(myfilter.wrapper,eval_data,eval_config)
filtered_examples=myfilter.recover_labels(myfilter.wrapper,new_examples,eval_config,recover_type='max_eachla')


path='/workspace/zhoujing/FewGLUE_dev32/augmented/WSC/t5_keep_0.3_default_sample0_beam10_augnum10wscaugtype_np_extra_train.jsonl'
from pet.tasks import WscProcessor
myprocessor=WscProcessor()
new_examples=myprocessor._create_examples(path,'train')
from genaug import confidence_filter
import imp
imp.reload(confidence_filter)
# myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='/workspace/zhoujing/pet_master/results/baseline/pet/wsc_albert_model/p2-i0')
myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='new_results/baseline/pet/wsc_albert_model/p2-i0')
import pet
eval_config = pet.EvalConfig(device='cuda:0', n_gpu=1, metrics='acc', per_gpu_eval_batch_size=1, decoding_strategy='default', priming=False)
myfilter.validate(myfilter.wrapper,new_examples[10:11],eval_config)
filtered_examples,num=myfilter.recover_labels(myfilter.wrapper,new_examples[10:20],eval_config,recover_type='max_prevla')



path='/workspace/zhoujing/FewGLUE_dev32/augmented/COPA/t5_flip_0.5_rand_iter_1_sample0_beam10_augnum10_train.jsonl'
from pet.tasks import CopaProcessor
myprocessor=CopaProcessor()
flip_examples=myprocessor._create_examples(path,'train')

path='/workspace/zhoujing/FewGLUE_dev32/augmented/COPA/t5_keep_0.5_rand_iter_1_sample0_beam10_augnum10_train.jsonl'
from pet.tasks import CopaProcessor
myprocessor=CopaProcessor()
keep_examples=myprocessor._create_examples(path,'train')


path='/workspace/zhoujing/FewGLUE_dev32/COPA/train.jsonl'
from pet.tasks import CopaProcessor
myprocessor=CopaProcessor()
train_examples=myprocessor._create_examples(path,'train')

new_examples=flip_examples+keep_examples
from genaug import confidence_filter
import imp
imp.reload(confidence_filter)
# myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='/workspace/zhoujing/pet_master/results/baseline/pet/copa_albert_model/p0-i0')
myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='new_results/baseline/pet/copa_albert_model/p0-i0')
import pet
eval_config = pet.EvalConfig(device='cuda:0', n_gpu=1, metrics='acc', per_gpu_eval_batch_size=1, decoding_strategy='default', priming=False)
# myfilter.validate(myfilter.wrapper,new_examples[10:11],eval_config)
filtered_examples,num=myfilter.recover_copa_labels(myfilter.wrapper,new_examples,eval_config,recover_type='max_eachla_sep')

imp.reload(confidence_filter)
newfilter=confidence_filter.Confidence_Filter()
filtered_examples,num=newfilter.recover_copa_labels(myfilter.wrapper,new_examples,eval_config,recover_type='max_eachla_sep',orig_data=train_examples)
'''
class Normal_Filter(object):
    '''
    from genaug import confidence_filter
    import imp
    imp.reload(confidence_filter)
    normalfilter=confidence_filter.Normal_Filter()
    filter_funcs=normalfilter.set_sequential_funcs(remove_final_punc=True,remove_question=True,keep_first_sentence=True)
    normalfilter.apply_filter('I am in Tsinghua. and you?')
    '''
    def __init__(self):
        pass
    
    def remove_final_punc(self,text):
        if text is None: return text
        return text.rstrip(string.punctuation)

    def remove_question(self,text):
        if text is None: return text
        new_text=text.lower()
        if new_text.startswith('did') or new_text.startswith('are') or new_text.startswith('is'):
            return None
        else:
            return text

    def keep_first_sentence(self,text):
        if text is None: return text
        return text[:len(re.split('\?|\.',text)[0])+1]

    def set_sequential_funcs(self,remove_final_punc=False,remove_question=False,keep_first_sentence=False):
        funcs=[]
        if remove_final_punc==True:
            funcs.append(self.remove_final_punc)
        if remove_question==True:
            funcs.append(self.remove_question)
        if keep_first_sentence==True:
            funcs.append(self.keep_first_sentence)
        self.funcs=funcs
        return funcs

    def apply_filter(self,text,funcs=None):
        if funcs is None:
            funcs=self.funcs
        for func in funcs:
            text=func(text)
        return text

class Confidence_Filter(object):
    def __init__(self,pattern_iter_output_dir=None,wrapper=None):
        assert pattern_iter_output_dir is None or wrapper is None
        self.wrappers=None
        self.wrapper=None
        if pattern_iter_output_dir is not None:
            self.wrapper=TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)
        if wrapper is not None:
            self.wrapper=wrapper

    def reload_wrapper(self,wrapper=None,pattern_iter_output_dir=None):
        if wrapper is not None:
            self.wrapper=wrapper
        else:
            if isinstance(pattern_iter_output_dir,list):
                self.wrappers=[]
                for path in pattern_iter_output_dir:
                    self.wrappers.append(TransformerModelWrapper.from_pretrained(path))
            else:
                self.wrapper=TransformerModelWrapper.from_pretrained(pattern_iter_output_dir)

    def validate(self,wrapper,eval_data,eval_config):
        if isinstance(wrapper,list):
            def merge_outputs(outputs,output):
                if outputs is None:
                    outputs=output
                else:
                    outputs['logits']=outputs['logits']+output['logits']
                return outputs
            total_data=list(itertools.chain.from_iterable(eval_data))
            outputs=None
            for wrp in wrapper:
                wrp.model.to(eval_config.device)
                output=wrp.eval(total_data,eval_config.device,per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
                    n_gpu=eval_config.n_gpu,decoding_strategy=eval_config.decoding_strategy,priming=eval_config.priming)
                outputs=merge_outputs(outputs,output)
                wrp.model.to('cpu')
                torch.cuda.empty_cache()
            outputs['logits']=outputs['logits']/len(wrapper)
            return outputs
        else:
            wrapper.model.to(eval_config.device)
            if isinstance(eval_data[0],list):
                total_data=list(itertools.chain.from_iterable(eval_data))
            else:
                total_data=eval_data
            output=wrapper.eval(total_data,eval_config.device,per_gpu_eval_batch_size=eval_config.per_gpu_eval_batch_size,
                    n_gpu=eval_config.n_gpu,decoding_strategy=eval_config.decoding_strategy,priming=eval_config.priming)
            torch.cuda.empty_cache()
            return output

    def rearrange_examples(self,examples):
        guids=[]
        for e in examples:
            if e.guid not in guids: guids.append(e.guid)
        guid_map={y:x for (x,y) in enumerate(guids)}
        new_examples=[[] for _ in range(len(guids))]
        for e in examples:
            new_examples[guid_map[e.guid]].append(e)
        return guids,new_examples

    def recover_copa_labels(self,wrapper,eval_data,eval_config,recover_type='max_eachla',orig_data=None):
        # select 'choice1' and 'choice2' separately
        example_num=len(eval_data)
        label_map=wrapper.preprocessor.label_map
        inverse_label_map={x:y for (y,x) in label_map.items()}
        label_num=len(label_map)
        return_examples=[];filtered_num=dict()
        guids,rearranged_examples=self.rearrange_examples(eval_data)
        if recover_type==('max_eachla_sep'):
            return_examples=[]
            for aug_examples in rearranged_examples:
                for la in range(label_num):
                    for question in ['effect','cause']:
                        orig_example=[e for e in orig_data if e.guid==aug_examples[0].guid][0]
                        examples=[e for e in aug_examples if (label_map[e.label]==la and e.meta['question']==question)]
                        if len(examples)==0: continue
                        tmp_examples=[]
                        # select the answer choice, keep label
                        for e in examples:
                            tmp_e=copy.deepcopy(e)
                            # orig_not_answer=orig_example.meta['choice1'] if orig_example.label=='1' else orig_example.meta['choice2']
                            if e.label=='0':
                                tmp_e.__dict__['meta']['choice2']=orig_example.meta['choice2']
                            else:
                                tmp_e.__dict__['meta']['choice1']=orig_example.meta['choice1']
                            tmp_examples.append(tmp_e)
                        logits=self.validate(wrapper,tmp_examples,eval_config)['logits']
                        logits=softmax(logits/10,axis=1)
                        print(logits)
                        max_idx1=-1
                        for (idx,logit) in enumerate(logits):
                            if (max_idx1==-1 or logit[la]>logits[max_idx1,la]):
                                max_idx1=idx
                        
                        tmp_examples=[tmp_examples[max_idx1]]
                        # select the answer choice, keep label
                        for e in examples:
                            tmp_e=copy.deepcopy(e)
                            tmp_e.__dict__['text_a']=examples[max_idx1].text_a
                            if e.label=='0':
                                tmp_e.__dict__['meta']['choice1']=examples[max_idx1].meta['choice1']
                                # tmp_e.__dict__['meta']['choice1']=orig_example.meta['choice1']
                            else:
                                tmp_e.__dict__['meta']['choice2']=examples[max_idx1].meta['choice2']
                                # tmp_e.__dict__['meta']['choice2']=orig_example.meta['choice2']
                            tmp_examples.append(tmp_e)
                        logits=self.validate(wrapper,tmp_examples,eval_config)['logits']
                        logits=softmax(logits/10,axis=1)
                        print(logits)
                        max_idx2=-1
                        for (idx,logit) in enumerate(logits):
                            if np.argmax(logit)==la and (max_idx2==-1 or logit[la]>logits[max_idx2,la]):
                                max_idx2=idx
                        if max_idx2!=-1:
                            new_example=copy.deepcopy(tmp_examples[max_idx2])
                            return_examples.append(new_example)
                            label_trans='{}-{} -> {}-{}'.format(new_example.meta['question'],new_example.orig_label,orig_example.meta['question'],new_example.label)
                            filtered_num.setdefault(label_trans,0)
                            filtered_num[label_trans]+=1
            return return_examples,filtered_num              
        else: 
            return NotImplemented

    def recover_labels(self,wrapper,eval_data,eval_config,recover_type='max_prevla',fixla_num=[[14,18],[14,18]],fixla_ratio=[[0.9,0.9],[0.9,0.9]],rmdup_num=1):
        # eval_data: [all_aug_examples]
        # recover_type: 
            # 'max_prevla': for each example, choose the most likely one whose label is preserved
            # 'max_eachla': for each example, choose the most likely one for each label if possible
            # 'max_otherla': for each example, choose the most likely one whose label is flipped
            # 'global_topk': choose examples whoce
            # 'global_topp': chooce examples whose confidence > topp
        example_num=len(eval_data)
        label_map=wrapper.preprocessor.label_map
        inverse_label_map={x:y for (y,x) in label_map.items()}
        label_num=len(label_map)
        return_examples=[];filtered_num=dict()
        guids,rearranged_examples=self.rearrange_examples(eval_data)
        if recover_type==('max_prevla'):
            for aug_examples in rearranged_examples:
                examples=[e for e in aug_examples if e.label==e.orig_label]
                if len(examples)==0: continue
                orig_la=label_map[examples[0].orig_label]
                la=orig_la
                logits=self.validate(wrapper,examples,eval_config)['logits']
                logits=softmax(logits/10,axis=1)
                # max_idx=np.argmax(logits[:,orig_la])
                max_idx=-1
                for (idx,logit) in enumerate(logits):
                    if np.argmax(logit)==la and (max_idx==-1 or logit[la]>logits[max_idx,la]):
                        max_idx=idx
                if max_idx!=-1:
                    return_examples.append(examples[max_idx])
                    label_trans='{} -> {}'.format(examples[max_idx].orig_label,examples[max_idx].label)
                    filtered_num.setdefault(label_trans,0)
                    filtered_num[label_trans]+=1
        elif recover_type==('max_prevla_comb'):
            for aug_examples in rearranged_examples:
                examples=aug_examples
                if len(examples)==0: continue
                orig_la=label_map[examples[0].orig_label]
                la=orig_la
                logits=self.validate(wrapper,examples,eval_config)['logits']
                logits=softmax(logits/10,axis=1)
                # max_idx=np.argmax(logits[:,orig_la])
                max_idx=-1
                for (idx,logit) in enumerate(logits):
                    if np.argmax(logit)==la and (max_idx==-1 or logit[la]>logits[max_idx,la]):
                        max_idx=idx
                if max_idx!=-1:
                    new_example=copy.deepcopy(examples[max_idx])
                    new_example.label=inverse_label_map[la]
                    return_examples.append(new_example)
                    label_trans='{} -> {}'.format(examples[max_idx].orig_label,examples[max_idx].label)
                    filtered_num.setdefault(label_trans,0)
                    filtered_num[label_trans]+=1        
        elif recover_type==('max_otherla'):
            for aug_examples in rearranged_examples:
                orig_la=label_map[aug_examples[0].orig_label]
                for la in range(label_num):
                    if la==orig_la: continue
                    examples=[e for e in aug_examples if label_map[e.label]==la]
                    if len(examples)==0: continue
                    logits=self.validate(wrapper,examples,eval_config)['logits']
                    logits=softmax(logits/10,axis=1)
                    max_idx=-1
                    for (idx,logit) in enumerate(logits):
                        if np.argmax(logit)==la and (max_idx==-1 or logit[la]>logits[max_idx,la]):
                            max_idx=idx
                    if max_idx!=-1:
                        return_examples.append(examples[max_idx])
                        label_trans='{} -> {}'.format(examples[0].orig_label,inverse_label_map[la])
                        filtered_num.setdefault(label_trans,0)
                        filtered_num[label_trans]+=1
        elif recover_type==('max_otherla_comb'):
            for aug_examples in rearranged_examples:
                orig_la=label_map[aug_examples[0].orig_label]
                examples=aug_examples
                if len(examples)==0: continue
                logits=self.validate(wrapper,examples,eval_config)['logits']
                logits=softmax(logits/10,axis=1)
                for la in range(label_num):
                    if la==orig_la: continue
                    max_idx=-1
                    for (idx,logit) in enumerate(logits):
                        if np.argmax(logit)==la and (max_idx==-1 or logit[la]>logits[max_idx,la]):
                            max_idx=idx
                    if max_idx!=-1:
                        new_example=copy.deepcopy(examples[max_idx])
                        new_example.label=inverse_label_map[la]
                        return_examples.append(new_example)
                        label_trans='{} -> {}'.format(examples[0].orig_label,inverse_label_map[la])
                        filtered_num.setdefault(label_trans,0)
                        filtered_num[label_trans]+=1
        elif recover_type==('max_eachla'): # We may flip the label according to the filter
            for examples in rearranged_examples:
                logits=self.validate(wrapper,examples,eval_config)['logits']
                logits=softmax(logits/10,axis=1)
                for la in range(label_num):
                    max_idx=-1
                    for (idx,logit) in enumerate(logits):
                        if np.argmax(logit)==la and (max_idx==-1 or logit[la]>logits[max_idx,la]):
                            max_idx=idx
                    if max_idx!=-1:
                    # max_idx=np.argmax(logits[:,la])
                    # if np.argmax(logits[max_idx])==la:
                        new_example=copy.deepcopy(examples[max_idx])
                        new_example.label=inverse_label_map[la]
                        return_examples.append(new_example)
                        label_trans='{} -> {}'.format(examples[0].orig_label,inverse_label_map[la])
                        filtered_num.setdefault(label_trans,0)
                        filtered_num[label_trans]+=1
        elif recover_type==('max_eachla_sep'):
            for aug_examples in rearranged_examples:
                for la in range(label_num):
                    examples=[e for e in aug_examples if label_map[e.label]==la]
                    if len(examples)==0: continue
                    logits=self.validate(wrapper,examples,eval_config)['logits']
                    logits=softmax(logits/10,axis=1)
                    max_idx=-1
                    for (idx,logit) in enumerate(logits):
                        if np.argmax(logit)==la and (max_idx==-1 or logit[la]>logits[max_idx,la]):
                            max_idx=idx
                    if max_idx!=-1:
                    # max_idx=np.argmax(logits[:,la])
                    # if np.argmax(logits[max_idx])==la:
                        return_examples.append(examples[max_idx])
                        label_trans='{} -> {}'.format(examples[0].orig_label,inverse_label_map[la])
                        filtered_num.setdefault(label_trans,0)
                        filtered_num[label_trans]+=1                    
        elif recover_type.startswith('global_topk'):   
            for orig_la in range(label_num):
                if 'sep' not in recover_type:
                    examples=[e for e in eval_data if (label_map[e.orig_label]==orig_la)]
                    if len(examples)==0: continue
                    logits=self.validate(wrapper,examples,eval_config)['logits']
                    logits=softmax(logits/10,axis=1)
                for new_la in range(label_num):
                    record_guids=defaultdict(int)
                    if 'sep' in recover_type:
                        examples=[e for e in eval_data if (label_map[e.orig_label]==orig_la and label_map[e.label]==new_la)]
                        if len(examples)==0: continue
                        logits=self.validate(wrapper,examples,eval_config)['logits']
                        logits=softmax(logits/10,axis=1)
                    aug_num=fixla_num[orig_la][new_la]
                    sortedindexs=np.argsort(logits[:,new_la])[::-1]
                    for k in range(aug_num):
                        if 'rmdup' in recover_type and record_guids[examples[sortedindexs[k]].guid]>=rmdup_num:
                            continue
                        new_example=copy.deepcopy(examples[sortedindexs[k]])
                        new_example.label=inverse_label_map[new_la]
                        return_examples.append(new_example)
                        label_trans='{} -> {}'.format(inverse_label_map[orig_la],inverse_label_map[new_la])
                        filtered_num.setdefault(label_trans,0)
                        filtered_num[label_trans]+=1
                        record_guids[new_example.guid]+=1   
        elif recover_type.startswith('global_topp'):
            for orig_la in range(label_num):
                if 'sep' not in recover_type:
                    examples=[e for e in eval_data if (label_map[e.orig_label]==orig_la)]
                    if len(examples)==0: continue
                    logits=self.validate(wrapper,examples,eval_config)['logits']
                    logits=softmax(logits,axis=1)
                for new_la in range(label_num):
                    record_guids=defaultdict(int)
                    if 'sep' in recover_type:
                        examples=[e for e in eval_data if (label_map[e.orig_label]==orig_la and label_map[e.label]==new_la)]
                        if len(examples)==0: continue
                        logits=self.validate(wrapper,examples,eval_config)['logits']
                        logits=softmax(logits,axis=1)
                    for (e,logit) in zip(examples,logits):
                        if 'rmdup' in recover_type and record_guids[e.guid]>=rmdup_num:
                            continue
                        if logit[new_la]>=fixla_ratio[orig_la][new_la]:
                            new_example=copy.deepcopy(e)
                            new_example.label=inverse_label_map[new_la]
                            return_examples.append(new_example)
                            # return_examples.append(e)
                            label_trans='{} -> {}'.format(inverse_label_map[orig_la],inverse_label_map[new_la])
                            filtered_num.setdefault(label_trans,0)
                            filtered_num[label_trans]+=1
                            record_guids[e.guid]+=1
        elif recover_type.startswith('deterministic_topk'):
            for orig_la in range(label_num):
                if 'sep' not in recover_type:
                    examples=[e for e in eval_data if (label_map[e.orig_label]==orig_la)]
                    if len(examples)==0: continue
                    logits=self.validate(wrapper,examples,eval_config)['logits']
                    logits=softmax(logits/10,axis=1)
                for new_la in range(label_num):
                    if 'sep' in recover_type:
                        examples=[e for e in eval_data if (label_map[e.orig_label]==orig_la and label_map[e.label]==new_la)]
                        if len(examples)==0: continue
                        logits=self.validate(wrapper,examples,eval_config)['logits']
                        logits=softmax(logits/10,axis=1)
                    aug_num=fixla_num[orig_la][new_la]
                    # prepare sorted grouped list
                    guids=[]
                    for e in examples:
                        if e.guid not in guids: guids.append(e.guid)
                    guid_map={y:x for (x,y) in enumerate(guids)}
                    new_examples=[[] for _ in range(len(guids))]
                    for (e,score) in zip(examples,logits[:,new_la]):
                        new_examples[guid_map[e.guid]].append((e,score))
                    for i in range(len(new_examples)):
                        new_examples[i]=sorted(new_examples[i],key=lambda x:x[1])[::-1]
                    # prepare sorted ungrouped list
                    sorted_ungrouped_examples=[]
                    for j in range(len(new_examples[0])):
                        tmp_examples=[]
                        for i in range(len(new_examples)):
                            tmp_examples.append(new_examples[i][j])
                        tmp_examples=sorted(tmp_examples,key=lambda x:x[1])[::-1]
                        sorted_ungrouped_examples+=tmp_examples
                    for (e,score) in sorted_ungrouped_examples[:aug_num]:    
                        new_example=copy.deepcopy(e)
                        new_example.label=inverse_label_map[new_la]
                        return_examples.append(new_example)
                        # return_examples.append(e)
                        label_trans='{} -> {}'.format(inverse_label_map[orig_la],inverse_label_map[new_la])
                        filtered_num.setdefault(label_trans,0)
                        filtered_num[label_trans]+=1
        return return_examples,filtered_num
        
    def del_finetuned_model(self):
        if self.wrappers is not None:
            for i in range(len(self.wrappers)):
                self.wrappers[i].model.cpu()
                self.wrappers[i].model=None
                self.wrappers[i]=None
            torch.cuda.empty_cache()
        else:
            self.wrapper.model.cpu()
            self.wrapper.model = None
            self.wrapper = None
            torch.cuda.empty_cache()
'''
path='/workspace/zhoujing/FewGLUE_dev32/augmented/MultiRC/t5_flip_0.1_default_sample0_beam10_augnum10_train.jsonl'
# path='/workspace/zhoujing/FewGLUE_dev32/augmented/MultiRC/t5_mlm_train.jsonl'
from pet.tasks import MultiRcProcessor
from pet import tasks
import imp
imp.reload(tasks)
myprocessor=tasks.MultiRcProcessor()
eval_data=myprocessor._create_examples(path,'aug')

# from genaug import confidence_filter
# import imp
# imp.reload(confidence_filter)
# myfilter=confidence_filter.Confidence_Filter()
# guids,new_examples=myfilter.rearrange_examples(eval_data)

import pet
imp.reload(confidence_filter)
myfilter=confidence_filter.Confidence_Filter(pattern_iter_output_dir='/workspace/zhoujing/data/pet_final/boolq_32_albert_model/p4-i0')
imp.reload(confidence_filter)
new_filter=confidence_filter.Confidence_Filter()
eval_config = pet.EvalConfig(device='cuda:0', n_gpu=1, metrics='acc', per_gpu_eval_batch_size=8, decoding_strategy='default', priming=False)
filtered_examples,filtered_num=new_filter.recover_labels(myfilter.wrapper,eval_data,eval_config,recover_type='global_topk_rmdup')
filtered_examples,filtered_num=new_filter.recover_labels(myfilter.wrapper,eval_data,eval_config,recover_type='global_topp_rmdup')
'''
               
