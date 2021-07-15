# FlipDA

This repository contains the official code for FlipDA.

We provide an automatic data augmentation method based on T5 and self-training by flipping the label. We evaluate it on FewGLUE, and improve its performance.

## Model
![image](https://github.com/zhouj8553/FlipDA/blob/main/img/model.png)

## Setup
Install and setup environment with requirements.txt.
The FewGLUE dataset is in the folder "data".

## Run Baseline
First, you should run the baseline
```Bash
bash scripts/run_pet.sh **task_name** 0 baseline
```

## Run baselines with augmented files
```Bash
bash scripts/run_pet.sh boolq 0 augmented_file_name
```

## Run FlipDA with augmented files
If the augmented_file_name has the corresponding version, we will load it. For example, if the filename is "t5_flip_0.5_default_sample0_beam1_augnum1" and we find "t5_keep_0.5_default_sample0_beam1_augnum10", we will load them both.

If you allow the model to update the labeled data by the T5-model, run the command as follows, where **augmented_file_name** is the augmented file name such as "t5_flip_0.5_default_sample0_beam1_augnum1".
```Bash
bash scripts/run_pet.sh boolq 0 genaug_**augmented_file_name**_filter_max_eachla
```

If you do not allow the model to update the labeled data by the T5-model, run the command as follows, where **augmented_file_name** is the augmented file name such as "t5_flip_0.5_default_sample0_beam1_augnum1".
```Bash
bash scripts/run_pet.sh boolq 0 genaug_**augmented_file_name**_filter_max_eachla_sep
```

Note that which command to choose is based on the relative power of the augmentation model and the classification model. If the augmentation model is accurate enough, choose the command with "sep". Otherwise, choose the first one.  

