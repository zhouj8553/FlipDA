import collections
import string
import re
from pet import tasks

def normalize_answer(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
		return re.sub(regex, ' ', text)
	def white_space_fix(text):
		return ' '.join(text.split())
	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)
	def lower(text):
		return text.lower()
	return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
	if not s: return []
	return normalize_answer(s).split()

def compute_f1(a_gold, a_pred):
	gold_toks = get_tokens(a_gold)
	pred_toks = get_tokens(a_pred)
	common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
	num_same = sum(common.values())
	if len(gold_toks) == 0 or len(pred_toks) == 0:
		# If either is no-answer, then F1 is 1 if they agree, 0 otherwise
		return int(gold_toks == pred_toks)
	if num_same == 0:
		return 0
	precision = 1.0 * num_same / len(pred_toks)
	recall = 1.0 * num_same / len(gold_toks)
	f1 = (2 * precision * recall) / (precision + recall)
	return f1

def read_file(filename):
	f= open(filename,"r")
	lines=f.readlines()
	ans=[]
	for line in lines:
		ans.append(eval(line))
	f.close()
	return ans

def get_max_ans(ans):
	aans={}
	for line in ans:
		max_val=-1000000
		text=''
		# print(line)
		for (choice,v) in line['choices'].items():
			if v>max_val:
				max_val=v
				text=choice
		aans[line['idx']]=text
	return aans

def get_evaluate_examples(filename='../../../data/FewGLUE/ReCoRD'):
	myprocessor=tasks.RecordProcessor()
	eval_examples=myprocessor.get_dev_examples(filename)
	return eval_examples


def cal_f1(ans,eval_examples):
	aans=get_max_ans(ans)
	f1s=[]
	for example in eval_examples:
		qid=example.meta['question_idx']
		targets=example.meta['answers']
		f_tmp=max([compute_f1(target,aans[qid]) for target in targets])
		f1s.append(f_tmp)
	return sum(f1s)/len(f1s)

def get_f1_from_file(target_filename,pred_filename):
	ans=read_file(pred_filename)
	eval_examples=get_evaluate_examples(target_filename)
	return cal_f1(ans,eval_examples)

# if __name__ == '__main__':
# 	f11=get_f1_from_file('../../../data/FewGLUE/ReCoRD','results/pet/record_32_model/p0-i0/predictions.jsonl')
# 	print('f1',f11)
# 	f12=get_f1_from_file('../../../data/FewGLUE/ReCoRD','results/pet/record_32_model/p0-i1/predictions.jsonl')
# 	print('f1',f12)
# 	f13=get_f1_from_file('../../../data/FewGLUE/ReCoRD','results/pet/record_32_model/p0-i2/predictions.jsonl')
# 	print('f1',f13)
# 	print((f11+f12+f13)/3)