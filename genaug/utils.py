from transformers import BasicTokenizer
basic_tokenizer=BasicTokenizer(do_lower_case=False)
def whitepiece_line_tokenizer(line):
    return basic_tokenizer.tokenize(line)

import nltk
def nltk_line_tokenizer(line):
    return nltk.word_tokenize(line)

import re
punc = '~`!#$%^&*()_+-=|\';":/.,?><~·！@#￥%……&*（）——+-=“：’；、。，？》《{}'
reg = "[^0-9A-Za-z\u4e00-\u9fa5]"

def removePunctuation(text):
    # text = re.sub(r'[{}]+'.format(punctuation),'',text)
    # text = (re.sub(punc, "",text)).replace('[','').replace(']','').replace(' ','')
    text=re.sub(reg, '', text)
    return text.strip()
    
import string
stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 
            'ours', 'ourselves', 'you', 'your', 'yours', 
            'yourself', 'yourselves', 'he', 'him', 'his', 
            'himself', 'she', 'her', 'hers', 'herself', 
            'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 
            'whom', 'this', 'that', 'these', 'those', 'am', 
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did',
            'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'of', 'at', 
            'by', 'for', 'with', 'about', 'against', 'between',
            'into', 'through', 'during', 'before', 'after', 
            'above', 'below', 'to', 'from', 'up', 'down', 'in',
            'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'here', 'there', 'when', 
            'where', 'why', 'how', 'all', 'any', 'both', 'each', 
            'few', 'more', 'most', 'other', 'some', 'such', 'no', 
            'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 
            'very', 's', 't', 'can', 'will', 'just', 'don', 
            'should', 'now']+[x for x in string.punctuation]

def find_all_nouns(lines,parallel=40):
    import spacy
    nlp = spacy.load("en")
    if isinstance(lines,list):
        ans=[]
        for line in lines:
            doc = nlp(line)
            for np in doc.noun_chunks:
                if len(np.text.split())==1 and np.text in stop_words:
                    continue
                ans.append(np.text)
    else:
        doc = nlp(lines)
        ans=[]
        for np in doc.noun_chunks:
            ans.append(np.text)
    return ans


def get_pps(doc):
    # import spacy
    # nlp=spacy.load('en')
    # doc=nlp('A short man in blue jeans is working in the kitchen.')
    # print(get_pps(doc))
    pps = [];pos=[]
    for (idx,token) in enumerate(doc):
        # Try this with other parts of speech for different subtrees.
        if token.pos_ == 'ADP':
            # import pdb
            # pdb.set_trace()
            pp = ' '.join([tok.orth_ for tok in token.subtree])
            for (i,tok) in enumerate(token.subtree):
            	if tok.orth_==str(token): start_pos=idx-i;break
            pps.append(pp)
            pos.append((start_pos,start_pos+len(pp.split())))
    return pps,pos


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
class SBERT():
    def __init__(self,):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def get_embeddings(self,sentences):
        return self.model.encode(sentences)

    def get_cos_similarity(self,sent_embs=[[0,1,2],[2,3,4]],tgt_emb=[5,6,7]):
        scores=[]
        for sent_emb in sent_embs:
            scores.append(cosine_similarity(np.array([sent_emb,tgt_emb]))[0,1])
        return scores

def softmax(x,temp=1):
    x=np.array(x)
    x_max=np.max(x)
    x=x-x_max
    x_exp=np.exp(x/temp)
    x_exp_sum=x_exp.sum()
    softmax=x_exp/x_exp_sum
    return softmax



SPECIAL_TOKENS={
    'bert':{
        'unk_token': '[UNK]',
        'sep_token': '[SEP]',
        'pad_token': '[PAD]',
        'cls_token': '[CLS]',
        'mask_token': '[MASK]'      
    },
    'albert':{
        'bos_token': '[CLS]',
        'eos_token': '[SEP]',
        'unk_token': '<unk>',
        'sep_token': '[SEP]',
        'pad_token': '<pad>',
        'cls_token': '[CLS]',
        'mask_token': '[MASK]'
    }
}

import json

def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as f:
	    dic = f.read()
    dic=eval(dic)
    return dic

def save_dict(dict,filename):
    with open(filename,'w') as f:
        f.write(str(dict))
    return

'''
from baseline_aug import utils
imp.reload(utils)
file_path='tmp/candidate_nouns.jsonl'
utils.save_dict(candidate_nouns,file_path)
nouns=utils.load_dict(file_path)
'''
