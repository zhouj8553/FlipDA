import torch
import numpy as np
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
from transformers import T5ForConditionalGeneration, T5Tokenizer

from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    # TemperatureLogitsWarper,
    # TopKLogitsWarper,
    # TopPLogitsWarper,
)

from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    StoppingCriteria,
    # validate_stopping_criteria,
)

T5_start_mask_token=32000
T5_end_mask_token=32099

class ForcedNoEOSTokenLogitsProcessor(LogitsProcessor):
    def __init__(self,encoder_input_ids: torch.LongTensor, eos_token_id: int): # (batch_size,...)
        self.target_blank_num=((encoder_input_ids>=T5_start_mask_token)&(encoder_input_ids<=T5_end_mask_token)).sum(dim=1)
        self.starts_with_extraid=((encoder_input_ids[:,0]>=T5_start_mask_token)&(encoder_input_ids[:,0]<=T5_end_mask_token)).int()
        self.batch_size = encoder_input_ids.shape[0]
        self.eos_token_id=eos_token_id
        self.pad_token_id=0

    def __call__(self,input_ids:torch.LongTensor,scores=torch.FloatTensor): #(batch_size*num_beams,...)
        num_hypos=scores.shape[0]
        num_beams=num_hypos//self.batch_size
        if input_ids.shape[1]<=1: return scores
        already_blank_num=(((input_ids>=T5_start_mask_token)&(input_ids<=T5_end_mask_token))).sum(dim=1)
        generated_extraid_first=((input_ids[:,1]>=T5_start_mask_token)&(input_ids[:,1]<=T5_end_mask_token)).int()
        for hypo_idx in range(num_hypos):
            batch_idx=hypo_idx//num_beams
            beam_idx=hypo_idx%num_beams
            if already_blank_num[hypo_idx]-generated_extraid_first[hypo_idx]+1<self.target_blank_num[batch_idx]: 
                next_extra_id=T5_end_mask_token-already_blank_num[batch_idx].item()
                scores[hypo_idx,next_extra_id]=torch.log(torch.exp(scores[hypo_idx,next_extra_id])+torch.exp(scores[hypo_idx,self.pad_token_id])+torch.exp(scores[hypo_idx,self.eos_token_id]))
                scores[hypo_idx,self.eos_token_id]=-float("inf")
                scores[hypo_idx,self.pad_token_id]=-float("inf")
        return scores

class ForcedNoSpanTokenLogitsProcessor(LogitsProcessor):
    def __init__(self,encoder_input_ids: torch.LongTensor, eos_token_id: int): # (batch_size,...)
        self.batch_size = encoder_input_ids.shape[0]
        self.eos_token_id=eos_token_id
        self.pad_token_id=0

    def __call__(self,input_ids:torch.LongTensor,scores=torch.FloatTensor): #(batch_size*num_beams,...)
        num_hypos=scores.shape[0]
        num_beams=num_hypos//self.batch_size
        for hypo_idx in range(num_hypos):
            batch_idx=hypo_idx//num_beams
            beam_idx=hypo_idx%num_beams
            if (input_ids[hypo_idx,-1]>=T5_start_mask_token and input_ids[hypo_idx,-1]<=T5_end_mask_token):  
                scores[hypo_idx,T5_start_mask_token:T5_end_mask_token+1]=-float("inf")
                scores[hypo_idx,self.eos_token_id]=-float("inf")
                scores[hypo_idx,self.pad_token_id]=-float("inf")
        return scores

class ForcedNoExtraTokenLogitsProcessor(LogitsProcessor):
    def __init__(self,encoder_input_ids: torch.LongTensor, eos_token_id: int):
        self.target_blank_num=((encoder_input_ids>=T5_start_mask_token)&(encoder_input_ids<=T5_end_mask_token)).sum(dim=1)
        self.starts_with_extraid=((encoder_input_ids[:,0]>=T5_start_mask_token)&(encoder_input_ids[:,0]<=T5_end_mask_token)).int()
        self.batch_size = encoder_input_ids.shape[0]
        self.eos_token_id=eos_token_id

    def __call__(self,input_ids:torch.LongTensor,scores=torch.FloatTensor):
        num_hypos=scores.shape[0]
        num_beams=num_hypos//self.batch_size
        if input_ids.shape[1]<=1: return scores
        already_blank_num=(((input_ids>=T5_start_mask_token)&(input_ids<=T5_end_mask_token))).sum(dim=1)
        generated_extraid_first=((input_ids[:,1]>=T5_start_mask_token)&(input_ids[:,1]<=T5_end_mask_token)).int()
        for hypo_idx in range(num_hypos):
            batch_idx=hypo_idx//num_beams
            beam_idx=hypo_idx%num_beams
            if already_blank_num[hypo_idx]-generated_extraid_first[hypo_idx]+1>=self.target_blank_num[batch_idx]:
                scores[hypo_idx,self.eos_token_id]=max(scores[hypo_idx,self.eos_token_id],scores[hypo_idx,T5_start_mask_token:T5_end_mask_token+1].max())
                scores[hypo_idx,T5_start_mask_token:T5_end_mask_token+1]=-float("inf")
        return scores

class ForcedStartTokenLogitsProcessor(LogitsProcessor):
    def __init__(self,encoder_input_ids: torch.LongTensor, eos_token_id: int):
        self.starts_with_extraid=((encoder_input_ids[:,0]>=T5_start_mask_token)&(encoder_input_ids[:,0]<=T5_end_mask_token))
        self.batch_size = encoder_input_ids.shape[0]
        self.eos_token_id=eos_token_id

    def __call__(self,input_ids:torch.LongTensor,scores=torch.FloatTensor):
        if input_ids.shape[1]!=1: return scores
        num_hypos=scores.shape[0]
        num_beams=num_hypos//self.batch_size
        num_tokens = scores.shape[1]
        for hypo_idx in range(num_hypos):
            batch_idx=hypo_idx//num_beams
            beam_idx=hypo_idx%num_beams
            # if True:
            if self.starts_with_extraid[batch_idx]==False:
                scores[hypo_idx,:]=-float("inf")
                scores[hypo_idx,T5_start_mask_token:T5_end_mask_token]=0
            else:
                scores[hypo_idx,T5_start_mask_token:T5_end_mask_token+1]=-float("inf")
        return scores

class EosCriteria(StoppingCriteria):
    def __init__(self,):
        self.eos_token_id=1

    def __call__(self,input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return (input_ids==self.eos_token_id).sum()==len(input_ids)

class T5_Blank(T5ForConditionalGeneration):
    def _get_stopping_criteria(
        self,
        max_length: Optional[int],
        max_time: Optional[float],
    ) -> StoppingCriteriaList:
        stopping_criteria = StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
        if max_time is not None:
            stopping_criteria.append(MaxTimeCriteria(max_time=max_time))
        stopping_criteria.append(EosCriteria())
        return stopping_criteria

    def _get_logits_processor(
        self,
        repetition_penalty: float,
        no_repeat_ngram_size: int,
        encoder_no_repeat_ngram_size: int,
        encoder_input_ids: torch.LongTensor,
        bad_words_ids: List[List[int]],
        min_length: int,
        max_length: int,
        eos_token_id: int,
        forced_bos_token_id: int,
        forced_eos_token_id: int,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]],
        num_beams: int,
        num_beam_groups: int,
        diversity_penalty: float,
        remove_invalid_values: bool,
    ) -> LogitsProcessorList:
        """
        This class returns a :obj:`~transformers.LogitsProcessorList` list object that contains all relevant
        :obj:`~transformers.LogitsProcessor` instances used to modify the scores of the language model head.
        """

        # init warp parameters
        repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
        )
        encoder_no_repeat_ngram_size = (
            encoder_no_repeat_ngram_size
            if encoder_no_repeat_ngram_size is not None
            else self.config.encoder_no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
        min_length = min_length if min_length is not None else self.config.min_length
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        diversity_penalty = diversity_penalty if diversity_penalty is not None else self.config.diversity_penalty
        forced_bos_token_id = (
            forced_bos_token_id if forced_bos_token_id is not None else self.config.forced_bos_token_id
        )
        forced_eos_token_id = (
            forced_eos_token_id if forced_eos_token_id is not None else self.config.forced_eos_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )
        # instantiate processors list
        processors = LogitsProcessorList()

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        if diversity_penalty is not None and diversity_penalty > 0.0:
            processors.append(
                HammingDiversityLogitsProcessor(
                    diversity_penalty=diversity_penalty, num_beams=num_beams, num_beam_groups=num_beam_groups
                )
            )
        if repetition_penalty is not None and repetition_penalty != 1.0:
            processors.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
        if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
            processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))
        if encoder_no_repeat_ngram_size is not None and encoder_no_repeat_ngram_size > 0:
            if self.config.is_encoder_decoder:
                processors.append(EncoderNoRepeatNGramLogitsProcessor(encoder_no_repeat_ngram_size, encoder_input_ids))
            else:
                raise ValueError(
                    "It's impossible to use `encoder_no_repeat_ngram_size` with decoder-only architecture"
                )
        if bad_words_ids is not None:
            processors.append(NoBadWordsLogitsProcessor(bad_words_ids, eos_token_id))
        processors.append(ForcedNoEOSTokenLogitsProcessor(encoder_input_ids,eos_token_id))
        processors.append(ForcedNoExtraTokenLogitsProcessor(encoder_input_ids,eos_token_id))
        if min_length is not None and eos_token_id is not None and min_length > -1:
            processors.append(MinLengthLogitsProcessor(min_length, eos_token_id))
        if prefix_allowed_tokens_fn is not None:
            processors.append(PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams // num_beam_groups))
        if forced_bos_token_id is not None:
            processors.append(ForcedBOSTokenLogitsProcessor(forced_bos_token_id))
        if forced_eos_token_id is not None:
            processors.append(ForcedEOSTokenLogitsProcessor(max_length, forced_eos_token_id))
        if remove_invalid_values is True:
            processors.append(InfNanRemoveLogitsProcessor())
        return processors

def init_model(model_name_or_path='t5-large'):
	tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
	tokenizer.sep_token = '</s>'
	model = T5_Blank.from_pretrained(model_name_or_path)
	model=model.cuda()
	model.eval()
	return tokenizer,model

class T5Aug():
    def __init__(self,model_path='t5-large',tokenizer=None,model=None):
        if tokenizer is not None and model is not None:
            self.tokenizer=tokenizer;self.model=model
        elif model_path is not None:
            self.tokenizer,self.model=init_model(model_path)

    def generate_blanks(self,strings_to_be_generated,
        max_length: Optional[int] = 512,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 15,
        top_p: Optional[float] = 0.5,
        repetition_penalty: Optional[float] = 2.5,
        bad_words_ids: Optional[Iterable[int]] = [[3], [19794], [22354]],
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = 0,
        eos_token_id: Optional[int] = 1,
        length_penalty: Optional[float] = 0.0,
        no_repeat_ngram_size: Optional[int] = 3,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = 1,
        max_time: Optional[float] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = False,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        **model_kwargs,
    ):
        pred_blanks=[];pred_texts=[]
        tokenizer=self.tokenizer
        eos_token_id = tokenizer._convert_token_to_id('</s>')
        pad_token_id = tokenizer._convert_token_to_id('<pad>')
        start_mask_token=tokenizer._convert_token_to_id('<extra_id_99>')
        end_mask_token=tokenizer._convert_token_to_id('<extra_id_0>')
        batch_size=10
        for batch_idx in range(int(np.ceil(len(strings_to_be_generated)/batch_size))):
            sentences=strings_to_be_generated[batch_idx*batch_size:(batch_idx+1)*batch_size]
            input_ids=tokenizer(sentences,return_tensors='pt',padding=True).input_ids.cuda()
            outputs=self.model.generate(input_ids,max_length=max_length,min_length=min_length,do_sample=do_sample,early_stopping=early_stopping,\
                num_beams=num_beams,temperature=temperature,top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty,bad_words_ids=bad_words_ids,\
                bos_token_id=bos_token_id,pad_token_id=pad_token_id,eos_token_id=eos_token_id,length_penalty=length_penalty,no_repeat_ngram_size=no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,num_return_sequences=num_return_sequences,decoder_start_token_id=decoder_start_token_id,
                use_cache=use_cache,num_beam_groups=num_beam_groups,diversity_penalty=diversity_penalty)
            for (b_id,input_id) in enumerate(input_ids):
                pred_text=[];result = []
                for item in outputs[b_id*num_return_sequences:(b_id+1)*num_return_sequences]:
                    result.append([]);blanks=[]
                    for token_id in item[1:]:
                        token_id=token_id.item()
                        if (token_id>=start_mask_token and token_id<=end_mask_token) or token_id==eos_token_id or token_id==pad_token_id:
                            blanks.append([])
                        else:
                            if len(blanks)==0: blanks.append([])
                            blanks[-1].append(token_id)
                    for blank in blanks:
                        result[-1].append(tokenizer.decode(blank))

                    current_blank=0;output_tokens=[]
                    for token in input_id:
                        token=token.item()
                        if token>=start_mask_token and token<=end_mask_token:
                            if current_blank<len(blanks):
                                output_tokens+=blanks[current_blank]
                            current_blank+=1
                        else:
                            if token not in [pad_token_id,eos_token_id]:
                                output_tokens.append(token)
                    pred_text.append(tokenizer.decode(output_tokens))
                pred_texts.append(pred_text)
                pred_blanks.append(result)
        return pred_texts,pred_blanks        
