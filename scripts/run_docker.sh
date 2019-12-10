#!/bin/bash

args=""

if [ "$MODE" != "" ]
then
    args+=" --mode $MODE"
fi

if [ "$DATASET_NAME" != "" ]
then
    args+=" --dataset_name $DATASET_NAME"
fi

if [ "$DATASET_PATH" != "" ]
then
    args+=" --dataset_path \"$DATASET_PATH\""
fi

if [ "$MAX_DATA" != "" ]
then
    args+=" --max_data $MAX_DATA"
fi

if [ "$INPUT" != "" ]
then
    args+=" --input $INPUT"
fi

if [ "$BATCH_SIZE" != "" ]
then
    args+=" --batch_size $BATCH_SIZE"
fi

if [ "$EVAL_SIZE" != "" ]
then
    args+=" --eval_size $EVAL_SIZE"
fi

if [ "$LEARNING_RATE" != "" ]
then
    args+=" --learning_rate $LEARNING_RATE"
fi

if [ "$EPOCHS" != "" ]
then
    args+=" --epochs $EPOCHS"
fi

if [ "$MODEL_DIR" != "" ]
then
    args+=" --model_dir \"$MODEL_DIR\""
fi

if [ "$STEPS_PER_LOG" != "" ]
then
    args+=" --steps_per_log $STEPS_PER_LOG"
fi

if [ "$STEPS_PER_CHECKPOINT" != "" ]
then
    args+=" --steps_per_checkpoint $STEPS_PER_CHECKPOINT"
fi

if [ "$CHECKPOINT" != "" ]
then
    args+=" --checkpoint \"$CHECKPOINT\""
fi

if [ "$TB_LOG_DIR" != "" ]
then
    tensorboard --logdir $TB_LOG_DIR &
    args+=" --tb_log_dir \"$TB_LOG_DIR\""
else
    tensorboard --logdir ./logs &
fi

if [ "$KEEP_TOP" != "" ]
then
    args+=" --keep_top $KEEP_TOP"
fi

if [ "$SHUFFLE_BUFFER_SIZE" != "" ]
then
    args+=" --shuffle_buffer_size $SHUFFLE_BUFFER_SIZE"
fi

if [ "$ENCODER_LAYERS" != "" ]
then
    args+=" --encoder_layers $ENCODER_LAYERS"
fi

if [ "$ENCODER_SIZE" != "" ]
then
    args+=" --encoder_size $ENCODER_SIZE"
fi

if [ "$PRED_NET_LAYERS" != "" ]
then
    args+=" --pred_net_layers $PRED_NET_LAYERS"
fi

if [ "$PRED_NET_SIZE" != "" ]
then
    args+=" --pred_net_size $PRED_NET_SIZE"
fi

if [ "$JOINT_NET_SIZE" != "" ]
then
    args+=" --joint_net_size $JOINT_NET_SIZE"
fi

if [ "$SOFTMAX_SIZE" != "" ]
then
    args+=" --softmax_size $SOFTMAX_SIZE"
fi

if [ "$TPU" != "" ]
then
    args+=" --tpu $TPU"
fi

eval "python run_rnnt.py $args"