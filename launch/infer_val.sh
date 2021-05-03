#!/bin/bash

#
# Arguments
#
OUTPUT_DIR=./results

# validation
FILELIST=val_cityscapes

# train set
#FILELIST=train_cityscapes

# VGG16
CONFIG=configs/deeplabv2_vgg16_train.yaml
# ResNet-101
#CONFIG=configs/deeplabv2_resnet101_train.yaml
# VGG16-FCN
#CONFIG=configs/fcn_vgg16_train.yaml

# Running configs
DATALOADER=cityscapes

#### Example ####
EXP=baselines
RUN_ID=vgg16fcn_synthia
SNAPSHOT=baseline_abn_e040

SNAPSHOT_PATH=snapshots/cityscapes/${EXP}/${RUN_ID}/${SNAPSHOT}.pth

SAVE_ID=${RUN_ID}_${SNAPSHOT}

#
# Code goes here
#
LISTNAME=`basename $FILELIST .txt`
SAVE_DIR=$OUTPUT_DIR/$EXP/$DATALOADER/$SAVE_ID/$LISTNAME
LOG_FILE=$OUTPUT_DIR/$EXP/$DATALOADER/$SAVE_ID/$LISTNAME.log

NUM_THREADS=12
set OMP_NUM_THREADS=$NUM_THREADS
export OMP_NUM_THREADS=$NUM_THREADS

CMD="python infer_val.py  --dataloader $DATALOADER \
                          --cfg $CONFIG \
                          --exp $EXP \
                          --run $RUN_ID \
                          --resume $SNAPSHOT_PATH \
                          --infer-list $FILELIST \
                          --mask-output-dir $SAVE_DIR"

if [ ! -d $SAVE_DIR ]; then
  echo "Creating directory: $SAVE_DIR"
  mkdir -p $SAVE_DIR
else
  echo "Saving to: $SAVE_DIR"
fi

echo $CMD
nohup $CMD > $LOG_FILE 2>&1 &

sleep 1
tail -f $LOG_FILE
