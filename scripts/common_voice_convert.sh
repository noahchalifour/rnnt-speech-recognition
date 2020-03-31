#!/bin/bash

OIFS="$IFS"
IFS=$'\n'

FORMAT=.mp3
DATA_DIR="$1"

mkdir -p $DATA_DIR

FILES=$(ls "$DATA_DIR" | grep $FORMAT)

for FILE in $FILES
do
    FILENAME="${FILE:0:${#FILE}-4}"
    ffmpeg -i $DATA_DIR/$FILE -acodec pcm_s16le -ac 1 -ar 16000 $DATA_DIR/$FILENAME.wav
    rm $DATA_DIR/$FILE
done

IFS="$OIFS"