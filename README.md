# FlipDA

This repository contains the official code for FlipDA.

We provide an automatic data augmentation method based on T5 and self-training by flipping the label. We evaluate it on FewGLUE, and improve its performance.

## Model
![image](https://github.com/zhouj8553/FlipDA/blob/main/img/model.png)

## Setup
Install and setup environment with requirements.txt.
The FewGLUE dataset is in the folder "data".

## Run Baseline
```Bash
bash scripts/run_pet.sh boolq 0 baseline
```

## Run baselines with augmented files
```Bash
bash scripts/run_pet.sh boolq 0 augmented_file_name
```

