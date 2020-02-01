#!/bin/bash

args=""

if [ "$MODE" != "" ]
then
    args+=" --mode $MODE"
fi

if [ "$DATA_DIR" != "" ]
then
    args+=" --data_dir $DATA_DIR"
fi

if [ "$TB_LOG_DIR" != "" ]
then
    tensorboard --logdir $TB_LOG_DIR &
    args+=" --tb_log_dir \"$TB_LOG_DIR\""
else
    tensorboard --logdir ./logs &
fi

if [ "$MODEL_DIR" != "" ]
then
    args+=" --model_dir $MODEL_DIR"
fi

if [ "$BATCH_SIZE" != "" ]
then
    args+=" --batch_size $BATCH_SIZE"
fi

if [ "$N_EPOCHS" != "" ]
then
    args+=" --n_epochs $N_EPOCHS"
fi

eval "python run_common_voice.py $args"