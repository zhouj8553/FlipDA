# FlipDA

This repository contains the official code for FlipDA.

We provide an automatic data augmentation method based on T5 and self-training by flipping the label. We evaluate it on FewGLUE, and improve its performance.

## Model

![image](https://github.com/zhouj8553/FlipDA/blob/main-v2/img/model.png)

## Setup

Install and setup environment with requirements.txt.
All our experiments are conducted on a single V100 with 32G VIDEO MEMORY.
The FewGLUE dataset is default placed in the folder _data_. The data could be downloaded in https://cloud.tsinghua.edu.cn/d/0d1a737d3b4f40a3b853/.

## Step1: Run Baseline

First, you should run the baseline. <task_name> could be "boolq", "rte", "cb", "copa", "wsc", "wic", and "multirc". <gpu_id> could be 0,1,2,..., according to the number of your gpu.

```bash
bash scripts/run_pet.sh <task_name> <gpu_id> baseline
```

For example, to reproduce the baseline, you can run the commands as follows:

```bash
bash scripts/run_pet.sh boolq 0 baseline
bash scripts/run_pet.sh rte 1 baseline
bash scripts/run_pet.sh cb 2 baseline
bash scripts/run_pet.sh multirc 3 baseline
bash scripts/run_pet.sh copa 4 baseline
bash scripts/run_pet.sh wsc 5 baseline
bash scripts/run_pet.sh wic 6 baseline
bash scripts/run_pet.sh record 7 baseline
```

If you run the command and shell as default, the results will be in _results/baseline/pet/<task_name>\_albert\_model/result_test.txt_.

## Step2: Produce augmented files

The code to generate augmented examples by T5 model is in _genaug/total_gen_aug.py_.

You could use the command as follows to generate augmented examples. <task_name> could be "BoolQ", "RTE", "CB", "COPA", "WSC", "WiC", "MultiRC", and "ReCoRD". <mask_ratio> could be arbitrary floating point number between 0 and 1, and in our experiments, we only tried 0.3, 0.5, and 0.8. <aug_type> could be "default" or "rand_iter_%d", where %d could be any integers. <label_type> could be "flip" or "keep". "do_sample" and "num_beams" controls the generation style (sample/greedy/beam search). <aug_num> denotes the number of augmented samples to be generated for each sample, in our experiments, we choose <aug_num> 10.

```bash
CUDA_VISIBLE_DEVICES=0 python -m genaug.total_gen_aug --task_name <task_name> --mask_ratio <mask_ratio> --aug_type <aug_type> --label_type <label_type> --do_sample --num_beams <num_beams> --aug_num <aug_num>
```

For example, to generate the augmented data we use in RTE, you could use the following command.

```bash
CUDA_VISIBLE_DEVICES=0 python -m genaug.total_gen_aug --task_name RTE --mask_ratio 0.5 --aug_type 'default' --label_type 'flip' --do_sample --num_beams 1  --aug_num 10
CUDA_VISIBLE_DEVICES=1 python -m genaug.total_gen_aug --task_name RTE --mask_ratio 0.5 --aug_type 'default' --label_type 'keep' --do_sample --num_beams 1  --aug_num 10

```

## Alternative: Run baselines with augmented files without classifier

If you want to add all the augmented data in the augmented files into the model (you do not want to change the label according to the trained classifier or filter the augmented examples), you could run the command as follows.

```bash
bash scripts/run_pet.sh boolq 0 <augmented_file_name>
```

## Step3: Run FlipDA with augmented files

If the <augmented_file_name> has the corresponding version, we will load it. For example, if the filename is "t5_flip_0.5_default_sample0_beam1_augnum10" and we find "t5_keep_0.5_default_sample0_beam1_augnum10", we will load them both.

If you allow the classifier to correct the label of the augmented data, run the command as follows, where <augmented_file_name> is the augmented file name such as "t5_flip_0.5_default_sample0_beam1_augnum10".

```bash
bash scripts/run_pet.sh boolq 0 genaug_<augmented_file_name>_filter_max_eachla
```

If you do not allow the classifier to correct the label of the augmented data, run the command as follows, where <augmented_file_name> is the augmented file name such as "t5_flip_0.5_default_sample0_beam1_augnum10".

```bash
bash scripts/run_pet.sh boolq 0 genaug_<augmented_file_name>_filter_max_eachla_sep
```

Note that which command to choose is based on the relative power of the augmentation model and the classification model. If the augmentation model is accurate enough, choosing the command with "sep" will be better. Otherwise, choose the first one. If you are not sure, just try them both.

To reproduce our result, command for AlBERT-xxlarge-v2 of FlipDA is:

```bash
bash scripts/run_pet.sh rte 0 genaug_t5_flip_0.5_default_sample1_beam1_augnum10_filter_max_eachla_sep
bash scripts/run_pet.sh boolq 0 genaug_t5_flip_0.3_default_sample1_beam1_augnum10_filter_max_eachla
bash scripts/run_pet.sh cb 0 genaug_t5_flip_0.5_default_sample1_beam1_augnum10_filter_max_eachla
bash scripts/run_pet.sh copa 0 genaug_t5_flip_0.8_default_sample0_beam10_augnum10_filter_max_eachla_sep
bash scripts/run_pet.sh wic 0 genaug_t5_flip_0.8_default_sample1_beam1_augnum10_filter_max_eachla_sep
bash scripts/run_pet.sh wsc 0 genaug_t5_keep_0.3_default_sample0_beam1_augnum10wscaugtype_extra_filter_max_prevla
bash scripts/run_pet.sh multirc 0 genaug_t5_flip_0.5_rand_iter_10_sample1_beam1_augnum10_filter_max_eachla_sep
bash scripts/run_pet.sh record 0 genaug_t5_flip_0.3_rand_iter_10_sample0_beam10_augnum10_filter_max_eachla
```

Note that we do not try all the hyperparameter combinations to save time and avoid overfitting, and our main contribution is proving that label flipping is useful for few-shot data augmentation. We will not be surprised if you get a better result with more careful implementation and hyperparameter selection.

## Citation

This paper will appear at ACL 2022 (Main Conference). Please cite us if FlipDA is useful in your work:

```
@misc{zhou2021flipda,
  title={FlipDA: Effective and Robust Data Augmentation for Few-Shot Learning}, 
  author={Jing Zhou and Yanan Zheng and Jie Tang and Jian Li and Zhilin Yang},
  year={2021},
  eprint={2108.06332},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```



