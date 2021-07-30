# FlipDA

This repository contains the official code for FlipDA.

We provide an automatic data augmentation method based on T5 and self-training by flipping the label. We evaluate it on FewGLUE, and improve its performance.

## Model
![image](https://github.com/zhouj8553/FlipDA/blob/main/img/model.png)

## Setup
Install and setup environment with requirements.txt.
The FewGLUE dataset is in the folder __data__. The data could be downloaded in https://cloud.tsinghua.edu.cn/d/0d1a737d3b4f40a3b853/.

## Run Baseline
First, you should run the baseline. <task_name> could be "boolq", "rte", "cb", "copa", "wsc", "wic", and "multirc". <gpu_id> could be 0,1,2,..., according to the number of your gpu.
```Bash
bash scripts/run_pet.sh <task_name> <gpu_id> baseline
```
For example, to reproduce the baseline, you can run the commands as follows:

```Bash
bash scripts/run_pet.sh boolq 0 baseline
bash scripts/run_pet.sh rte 1 baseline
bash scripts/run_pet.sh cb 2 baseline
bash scripts/run_pet.sh multirc 3 baseline
bash scripts/run_pet.sh copa 4 baseline
bash scripts/run_pet.sh wsc 5 baseline"
bash scripts/run_pet.sh wic 6 baseline
```

If you run the command and shell as default, the results will be in _results/baseline/pet/<task_name>\_albert\_model/result_test.txt_.

## Produce augmented files
The code to generate augmented examples by T5 model is in _genaug/total_gen_aug.py_.

You could use the command as follows to generate augmented examples. <task_name> could be "BoolQ", "RTE", "CB", "COPA", "WSC", "WiC", and "MultiRC". <mask_ratio> could be all float between 0 and 1, and in our experiments, we only tried 0.3, 0.5, and 0.8. <aug_type> could be "default" or "rand_iter_%d", where %d could be any integers. <label_type> could be "flip" or "keep". "do_sample" and "num_beams" controls the generation style (sample/greedy/beam search). <aug_num> denotes the number of sentences to be generated, in our experiments, we choose <aug_num> 10.

```Bash
CUDA_VISIBLE_DEVICES=0 python -m genaug.total_gen_aug --task_name <task_name> --mask_ratio <mask_ratio> --aug_type <aug_type> --label_type <label_type> --do_sample --num_beams <num_beams> --aug_num <aug_num>
```

For example, to generate the augmented data we use in RTE, you could use the following command.
```Bash
CUDA_VISIBLE_DEVICES=0 python -m genaug.total_gen_aug --task_name RTE --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10
CUDA_VISIBLE_DEVICES=1 python -m genaug.total_gen_aug --task_name RTE --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --do_sample --num_beams 1  --aug_num 10

```

## Run baselines with augmented files without classifier
If you want to add all the augmented data in the augmented files into the model (you do not want to change the label according to the trained classifier or filter the augmented examples), you could run the command as follows.
```Bash
bash scripts/run_pet.sh boolq 0 <augmented_file_name>
```

## Run FlipDA with augmented files
If the augmented_file_name has the corresponding version, we will load it. For example, if the filename is "t5_flip_0.5_default_sample0_beam1_augnum1" and we find "t5_keep_0.5_default_sample0_beam1_augnum10", we will load them both.

If you allow the model to update the labeled data by the T5-model, run the command as follows, where <augmented_file_name> is the augmented file name such as "t5_flip_0.5_default_sample0_beam1_augnum1".
```Bash
bash scripts/run_pet.sh boolq 0 genaug_<augmented_file_name>_filter_max_eachla
```

If you do not allow the model to update the labeled data by the T5-model, run the command as follows, where <augmented_file_name> is the augmented file name such as "t5_flip_0.5_default_sample0_beam1_augnum10".
```Bash
bash scripts/run_pet.sh boolq 0 genaug_<augmented_file_name>_filter_max_eachla_sep
```

Note that which command to choose is based on the relative power of the augmentation model and the classification model. If the augmentation model is accurate enough, choose the command with "sep". Otherwise, choose the first one.  

To reproduce our result, command for AlBERT-xxlarge-v2 of FlipDA is:
```Bash
bash scripts/run_pet.sh rte 0 genaug_t5_flip_0.5_default_sample1_beam1_augnum10_filter_max_eachla_sep
bash scripts/run_pet.sh boolq 0 genaug_t5_flip_0.3_default_sample1_beam1_augnum10_filter_max_eachla
bash scripts/run_pet.sh cb 0 genaug_t5_flip_0.5_default_sample1_beam1_augnum10_filter_max_eachla
bash scripts/run_pet.sh copa 0 genaug_t5_flip_0.8_default_sample0_beam10_augnum10_filter_max_eachla_sep
bash scripts/run_pet.sh wic 0 genaug_t5_flip_0.8_default_sample1_beam1_augnum10_filter_max_eachla_sep
bash scripts/run_pet.sh wsc 0 genaug_t5_keep_0.3_default_sample0_beam1_augnum10wscaugtype_extra_filter_max_prevla
bash scripts/run_pet.sh multirc 0 genaug_t5_flip_0.5_rand_iter_10_sample1_beam1_augnum10_filter_max_eachla_sep
```
Note that we do not try all the hyperparameter combinations, and our main contribution is to prove that label flipping is useful for few-shot data augmentation. We will not be surprised if you get a better result with more careful implementation and hyperparameter selection.


## Citation
Please cite us if FlipDA is useful in your work:
```Bash

```
