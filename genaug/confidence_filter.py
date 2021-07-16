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

class Normal_Filter(object):
    '''
    from genaug import confidence_filter
    normalfilter=confidence_filter.Normal_Filter()
    filter_funcs=normalfilter.set_sequential_funcs(remove_final_punc=True,remove_question=True,keep_first_sentence=True)
    normalfilter.apply_filter('I am a student. and you?')
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


    def recover_labels(self,wrapper,eval_data,eval_config,recover_type='max_prevla',fixla_num=[[14,18],[14,18]],fixla_ratio=[[0.9,0.9],[0.9,0.9]],rmdup_num=1):
        # eval_data: [all_aug_examples]
        # recover_type: 
            # 'max_prevla': for each example, choose the most likely one whose label is preserved
            # 'max_eachla': for each example, choose the most likely one for each label if possible
            # 'max_otherla': for each example, choose the most likely one whose label is flipped
            # 'global_topk': choose examples who are among the topk confident
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
        elif recover_type==('believe_cls'):
            logits=self.validate(wrapper,eval_data,eval_config)['logits']
            for (e,logit) in zip(eval_data,logits):
                orig_la=label_map[e.orig_label]
                new_la=np.argmax(logit)
                new_example=copy.deepcopy(e)
                new_example.label=inverse_label_map[new_la]
                return_examples.append(new_example)
                # return_examples.append(e)
                label_trans='{} -> {}'.format(inverse_label_map[orig_la],inverse_label_map[new_la])
                filtered_num.setdefault(label_trans,0)
                filtered_num[label_trans]+=1
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

       
