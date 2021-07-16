import numpy as np
import random
from baseline_aug.utils import *
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

'''
from baseline_aug import knn_replacement
import imp
imp.reload(knn_replacement)
myaug=knn_replacement.KNNAugment()
myaug.augment("Russia's natural gas monopoly Gazprom has also shown interest in acquiring a sizable stake in the Uzbek pipeline monopoly.")
'''
class KNNAugment(object):
    def __init__(self,):
        glove_path="/workspace/zhoujing/data/embedding/glove.6B.300d.txt"
        self.emb_norm, self.vocab, self.ids_to_tokens = prepare_embedding_retrieval(glove_path)

    def _word_distance(self, word):
        if word not in self.vocab.keys():
            return []
        word_idx = self.vocab[word]
        word_emb = self.emb_norm[word_idx]

        dist = np.dot(self.emb_norm, word_emb.T)
        dist[word_idx] = -np.Inf

        candidate_ids = np.argsort(-dist)[:self.M]
        return [self.ids_to_tokens[idx] for idx in candidate_ids][:self.M]

    def augment(self,text, M=15, p=0.1):
        self.M=M 
        words=whitepiece_line_tokenizer(text.lower())
        new_words=[]
        for word in words:
            if random.random()<p:
                candidate_words = self._word_distance(word)
                if len(candidate_words)==0:
                    new_words.append(word)
                else:
                    candidate_word = random.choice(candidate_words)
                    new_words.append(candidate_word)
            else:
                new_words.append(word)
        return ' '.join(new_words)

