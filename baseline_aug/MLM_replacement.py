import numpy as np
import random
from baseline_aug.utils import *
from transformers import pipeline,AlbertTokenizer
class MLMAugment(object):
    def __init__(self,):
        self.unmasker=pipeline('fill-mask',model='/workspace/zhoujing/data/checkpoints/albert-xxlarge-v2',device=0)
        self.tokenizer=AlbertTokenizer.from_pretrained('/workspace/zhoujing/data/checkpoints/albert-xxlarge-v2')

    def augment(self,text,p=0.1):
        words=whitepiece_line_tokenizer(text)
        text=' '.join(words)
        token_ids=self.tokenizer(text).input_ids
        # print(len(token_ids))
        if len(token_ids)>512:
            print('{} out of length'.format(len(token_ids)))
            text=self.tokenizer.decode(token_ids[:512]).replace('[CLS]','').replace('[SEP]','').strip()
        if len(words)==0: return text
        cnt=0
        mask_idxs=np.random.choice(len(words), size=max(int(len(words)*p),1), replace=False)
        for idx in mask_idxs:
            words[idx]='[MASK]'
            new_text=' '.join(words)
            token_ids=self.tokenizer(new_text).input_ids
            # print(len(token_ids))
            if len(token_ids)>512:
                print('{} out of length'.format(len(token_ids)))
                new_text=self.tokenizer.decode(token_ids[:512]).replace('[CLS]','').replace('[SEP]','').strip()
                if '[MASK]' not in new_text: break
            new_token=self.unmasker(new_text)[0]['token_str']
            words[idx]=new_token
        return ' '.join(words)

'''
from baseline_aug import MLM_replacement
import imp
imp.reload(MLM_replacement)
mlm_aug=MLM_replacement.MLMAugment()
aug_func=mlm_aug.augment
aug_func_name='mlm'
aug_kwargs={"p":0.1}
aug_num=10
aug_func("Russia's natural gas monopoly Gazprom has also shown interest in acquiring a sizable stake in the Uzbek pipeline monopoly",**aug_kwargs)


from transformers import pipeline,AlbertTokenizer
tokenizer=AlbertTokenizer.from_pretrained('/workspace/zhoujing/data/checkpoints/albert-xxlarge-v2')
text="Russia's natural gas monopoly Gazprom has also shown interest in acquiring a sizable stake in the Uzbek pipeline monopoly"
tokenizer.decode(tokenizer(text).input_ids)
'''