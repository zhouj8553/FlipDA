TASK=$1
device=$2
search_type=$3


METHOD='pet'
MODEL_TYPE='albert'

DATA_ROOT='data/FewGLUE_dev32/'
MODEL_NAME_OR_PATH="albert-xxlarge-v2"


echo Running iPET with the following parameters:
echo ------------------------------
echo TASK         = "$TASK"
echo METHOD       = "$METHOD"
echo MODEL_TYPE   = "$MODEL_TYPE"
echo device       = "$device"
echo DATA_ROOT    = "$DATA_ROOT"
echo MODEL_NAME_OR_PATH     = "$MODEL_NAME_OR_PATH"
echo ------------------------------




OUTPUT_DIR=results/${search_type}/${METHOD}/${TASK}_${MODEL_TYPE}_model

TRAIN_BATCH_SIZE=8
ACCU=2
SEQ_LENGTH=256


if [ $TASK = "wic" ]; then
  PATTERN_IDS="0 1 2"
  DATA_DIR=${DATA_ROOT}WiC
elif [ $TASK = "rte" ]; then
  # PATTERN_IDS="0 1 2 3 4"
  PATTERN_IDS="0 1 2 3"
  DATA_DIR=${DATA_ROOT}RTE
elif [ $TASK = "cb" ]; then
  # PATTERN_IDS="0 1 2 3 4"
  PATTERN_IDS="0 1 2 3"
  DATA_DIR=${DATA_ROOT}CB
elif [ $TASK = "wsc" ]; then
  PATTERN_IDS="0 1 2"
  DATA_DIR=${DATA_ROOT}WSC
  TRAIN_BATCH_SIZE=4
  ACCU=4
  SEQ_LENGTH=128
elif [ $TASK = "boolq" ]; then
  PATTERN_IDS="0 1 2 3 4 5"
  DATA_DIR=${DATA_ROOT}BoolQ
elif [ $TASK = "copa" ]; then
  PATTERN_IDS="0 1"
  DATA_DIR=${DATA_ROOT}COPA
  TRAIN_BATCH_SIZE=4
  ACCU=4
  SEQ_LENGTH=96
elif [ $TASK = "multirc" ]; then
  # PATTERN_IDS="0 1 2 3"
  PATTERN_IDS="0 1 2"
  DATA_DIR=${DATA_ROOT}MultiRC
  TRAIN_BATCH_SIZE=4
  ACCU=4
  SEQ_LENGTH=512
elif [ $TASK = "record" ]; then
  PATTERN_IDS="0"
  DATA_DIR=${DATA_ROOT}ReCoRD
  TRAIN_BATCH_SIZE=1
  ACCU=16
  SEQ_LENGTH=512
else
  echo "Task " $TASK " is not supported by this script" 1>&2
  exit 1
fi


if [[ $TASK = "record" || $TASK = "wsc" || $TASK = "copa" ]]; then
  echo "type1" $TASK
  CUDA_VISIBLE_DEVICES=$device nohup python3 cli.py \
  --method $METHOD \
  --pattern_ids $PATTERN_IDS \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --pet_per_gpu_eval_batch_size 1 \
  --pet_per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
  --pet_gradient_accumulation_steps $ACCU \
  --pet_max_steps 250 \
  --pet_max_seq_length $SEQ_LENGTH \
  --pet_repetitions 3 \
  --no_distillation \
  --search_type $search_type >myout_${METHOD}_${MODEL_TYPE}_${TASK}_${search_type}.file 2>&1 &
elif [[ $TASK = "rte" || $TASK = "cb" || $TASK = 'boolq' || $TASK = 'wic' || $TASK = 'multirc' ]]; then
  echo "type2" $TASK
  CUDA_VISIBLE_DEVICES=$device nohup python3 cli.py \
  --method $METHOD \
  --pattern_ids $PATTERN_IDS \
  --data_dir $DATA_DIR \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL_NAME_OR_PATH \
  --task_name $TASK \
  --output_dir $OUTPUT_DIR \
  --do_train \
  --do_eval \
  --pet_per_gpu_eval_batch_size 32 \
  --pet_per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
  --pet_gradient_accumulation_steps $ACCU \
  --pet_max_steps 250 \
  --pet_max_seq_length $SEQ_LENGTH \
  --pet_repetitions 3 \
  --no_distillation \
  --search_type $search_type >myout_${METHOD}_${MODEL_TYPE}_${TASK}_${search_type}.file 2>&1 &
fi

