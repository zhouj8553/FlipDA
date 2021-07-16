# https://github.com/huawei-noah/Pretrained-Language-Model/blob/1bb9b3e49b05279a4ba5cbe805ac81c803f94981/TinyBERT/data_augmentation.py#L123
# if it is in the format 'x ##y ##z': then use bert to predict it, else use glove
import random
import sys
import os
import unicodedata
import re
import logging
import csv
import argparse

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.enabled=False

import torch
import numpy as np

from transformers import BertTokenizer, BertForMaskedLM

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

StopWordsList = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                 'they', 'them', 'their', 'theirs', 'themselves', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
                 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                 'there', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've',
                 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
                 "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
                 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", "'s", "'re"]


def strip_accents(text):
    """
    Strip accents from input String.
    :param text: The input string.
    :type text: String.
    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError):
        # unicode is a default on python 3
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)


# valid string only includes al
def _is_valid(string):
    return True if not re.search('[^a-z]', string) else False


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(unicode(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


def prepare_embedding_retrieval(glove_file, vocab_size=100000):
    cnt = 0
    words = []
    embeddings = {}

    # only read first 100,000 words for fast retrieval
    with open(glove_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            items = line.strip().split()
            words.append(items[0])
            embeddings[items[0]] = [float(x) for x in items[1:]]

            cnt += 1
            if cnt == vocab_size:
                break

    vocab = {w: idx for idx, w in enumerate(words)}
    ids_to_tokens = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(embeddings[ids_to_tokens[0]])
    emb_matrix = np.zeros((vocab_size, vector_dim))
    for word, v in embeddings.items():
        if word == '<unk>':
            continue
        emb_matrix[vocab[word], :] = v

    # normalize each word vector
    d = (np.sum(emb_matrix ** 2, 1) ** 0.5)
    emb_norm = (emb_matrix.T / d).T
    return emb_norm, vocab, ids_to_tokens


class DataAugmentor(object):
    def __init__(self, model, tokenizer, emb_norm, vocab, ids_to_tokens, M, N, p):
        self.model = model
        self.tokenizer = tokenizer
        self.emb_norm = emb_norm
        self.vocab = vocab
        self.ids_to_tokens = ids_to_tokens
        self.M = M
        self.N = N
        self.p = p

    def _word_distance(self, word):
        if word not in self.vocab.keys():
            return []
        word_idx = self.vocab[word]
        word_emb = self.emb_norm[word_idx]

        dist = np.dot(self.emb_norm, word_emb.T)
        dist[word_idx] = -np.Inf

        candidate_ids = np.argsort(-dist)[:self.M]
        return [self.ids_to_tokens[idx] for idx in candidate_ids][:self.M]

    def _masked_language_model(self, sent, word_pieces, mask_id):
        tokenized_text = self.tokenizer.tokenize(sent)
        tokenized_text = ['[CLS]'] + tokenized_text
        tokenized_len = len(tokenized_text)

        tokenized_text = word_pieces + ['[SEP]'] + tokenized_text[1:] + ['[SEP]']

        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:512]

        token_ids = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if tokenized_len+1>512: #jing
            segments_ids=[0]*512
        else:
            segments_ids = [0] * (tokenized_len + 1) + [1] * (len(tokenized_text) - tokenized_len - 1)

        tokens_tensor = torch.tensor([token_ids]).to(device)
        segments_tensor = torch.tensor([segments_ids]).to(device)

        self.model.to(device)

        outputs = self.model(tokens_tensor, segments_tensor)
        # import pdb
        # pdb.set_trace()
        word_candidates = torch.argsort(outputs.logits[0, mask_id], descending=True)[:self.M].tolist()
        word_candidates = self.tokenizer.convert_ids_to_tokens(word_candidates)

        return list(filter(lambda x: x.find("##"), word_candidates))

    def _word_augment(self, sentence, mask_token_idx, mask_token):
        word_pieces = self.tokenizer.tokenize(sentence)
        word_pieces = ['[CLS]'] + word_pieces
        tokenized_len = len(word_pieces)

        token_idx = -1
        for i in range(1, tokenized_len):
            if "##" not in word_pieces[i]:
                token_idx = token_idx + 1
                if token_idx < mask_token_idx:
                    word_piece_ids = []
                elif token_idx == mask_token_idx:
                    word_piece_ids = [i]
                else:
                    break
            else:
                word_piece_ids.append(i)
            if i>=512: return [mask_token] #jing

        if len(word_piece_ids) == 1:
            word_pieces[word_piece_ids[0]] = '[MASK]'
            candidate_words = self._masked_language_model(
                sentence, word_pieces, word_piece_ids[0])
        elif len(word_piece_ids) > 1:
            candidate_words = self._word_distance(mask_token)
        else:
            logger.info("invalid input sentence!")
        
        if len(candidate_words)==0:
            candidate_words.append(mask_token)

        return candidate_words

    def augment(self, sent):
        candidate_sents = [sent]

        tokens = self.tokenizer.basic_tokenizer.tokenize(sent)
        candidate_words = {}
        for (idx, word) in enumerate(tokens):
            if _is_valid(word) and word not in StopWordsList:
                candidate_words[idx] = self._word_augment(sent, idx, word)
        logger.info(candidate_words)
        cnt = 0
        while cnt < self.N:
            new_sent = list(tokens)
            for idx in candidate_words.keys():
                candidate_word = random.choice(candidate_words[idx])

                x = random.random()
                if x < self.p:
                    new_sent[idx] = candidate_word

            if " ".join(new_sent) not in candidate_sents:
                candidate_sents.append(' '.join(new_sent))
            cnt += 1

        return candidate_sents


# class ArgumentProcessor(object):
#     def __init__(self,augmentor):
#         self.augmentor=augmentor

#     def read_augment_write(self,examples):
#         for e in examples:
#             sent=e.text_a;sent=e.text_b
#             augmented_sents=self.augmentor.augment(sent)

class Augment(object):
    def __init__(self,M=15,N=20,p=0.4):
        from transformers import BertTokenizer,BertForMaskedLM
        model_name_or_path='bert-base-uncased'
        tokenizer=BertTokenizer.from_pretrained(model_name_or_path)
        model = BertForMaskedLM.from_pretrained(model_name_or_path)

        glove_path="/workspace/zhoujing/data/embedding/glove.6B.300d.txt"
        emb_norm, vocab, ids_to_tokens = prepare_embedding_retrieval(glove_path)
        self.augmentor=DataAugmentor(model,tokenizer,emb_norm,vocab,ids_to_tokens,M=M,N=N,p=p)
        set_seed(1)

    def augment_sentences(self,lines):
        if isinstance(lines,list):
            tmp=[]
            for line in lines:
                tmp.append(self.augmentor.augment(line))
            return tmp
        else:
            return self.augmentor.augment(lines)

'''
from transformers import AlbertTokenizer,AlbertForMaskedLM
model_name_or_path='albert-xxlarge-v2'
tokenizer=AlbertTokenizer.from_pretrained(model_name_or_path)
model = AlbertForMaskedLM.from_pretrained(model_name_or_path)


from transformers import BertTokenizer,BertForMaskedLM
model_name_or_path='bert-base-uncased'
tokenizer=BertTokenizer.from_pretrained(model_name_or_path)
model = BertForMaskedLM.from_pretrained(model_name_or_path)

from baseline_aug import tinybert
glove_path="/workspace/zhoujing/data/embedding/glove.6B.300d.txt"
emb_norm, vocab, ids_to_tokens = prepare_embedding_retrieval(glove_path)

import imp
from baseline_aug import tinybert
imp.reload(tinybert)
from baseline_aug.tinybert import *
mydataaugmentor=tinybert.DataAugmentor(model,tokenizer,emb_norm,vocab,ids_to_tokens,M=15,N=20,p=0.4)
text="Officials claimed they were backed by influential members of the Santa Cruz business community of Croatian descent. The security vice-minister, Marcos Farfan, said that police have surveillance photographs of Mr Dwyer at various public events attended by Mr Morales, including a peasant rally near Santa Cruz and a visit to naval installations on Lake Titicaca. Mr Farfan said that Mr Dwyer was \"following\" Mr Morales and other officials as part of the preparations for the \"assassination plot\". He added that police experts are analysing contents reportedly found in computers taken from the rooms in which the men were killed."
ans=mydataaugmentor.augment(text)
'''



