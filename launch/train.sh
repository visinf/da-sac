#!/bin/bash

# logs/snapshots will be saved to
#   [logs|snapshots]/$DS/$EXP/$EXP_ID
# modify as needed
DS=cityscapes
EXP=v1_vgg
EXP_ID=01
LOG_DIR=logs/${DS}/${EXP}

# use pre-computed IS weights
# set to false if IS_WEIGHTS should be used instead
IS_WEIGHTS_DEFAULT=true

# if training with consistency
#   1) set the path to the baseline snapshot
#RESUME="snapshots/cityscapes/baselines/vgg16_gta/baseline_abn_e115.pth"

#   2) set the path to importance weights (optional)
#   if not set and not in baseline mode, it will attempt
#   to find default weights (provided for download) in data/.
#   if not found, it will use uniform sampling.
#IS_WEIGHTS="data/cs_weights_vgg16_gta.data"

#
# The rest will be handled from here
#

# defined by the arguments
SRC=$1
NET=$2
BASE=$3

case $SRC in
  gta)
  echo "Source: GTA5"
  TASK="TRAIN.TASK train_game_9K"
  ;;
  synthia)
  echo "Source: SYNTHIA"
  TASK="TRAIN.TASK train_synthia_9K VAL.IGNORE_CLASS 9,14,16"
  ;;
  *)
  echo "Source '$SRC' not supported. Choose from [gta|synthia]."
  exit 1
  ;;
esac

# net [resnet101|vgg16|vgg16fcn]
case $NET in
  resnet101)
  echo "Network: ResNet-101/DeepLabv2"
  if [ "$BASE" = "base" ]; then
    CFG=configs/deeplabv2_resnet101.yaml
  else
    CFG=configs/deeplabv2_resnet101_train.yaml
  fi
  ;;
  vgg16)
  echo "Network: VGG-16/DeepLabv2"
  if [ "$BASE" = "base" ]; then
    CFG=configs/deeplabv2_vgg16.yaml
  else
    CFG=configs/deeplabv2_vgg16_train.yaml
  fi
  ;;
  vgg16fcn)
  echo "Network: VGG-16/FCN"
  if [ "$BASE" = "base" ]; then
    CFG=configs/fcn_vgg16.yaml
  else
    CFG=configs/fcn_vgg16_train.yaml
  fi
  ;;
  *)
  echo "Network '$NET'. Choose from [resnet101|vgg16|vgg16fcn]."
esac

echo "Config: $CFG"
EXTRA="$TASK"

IS_WEIGHTS_="data/cs_weights_${NET}_${SRC}.data"
if [ "$IS_WEIGHTS_DEFAULT" = true ]; then
  IS_WEIGHTS=$IS_WEIGHTS_
fi

if [ "$BASE" = "base" ]; then
  echo "Training the ABN baseline"
  EXTRA="MODEL.BASELINE True $EXTRA"
  EXP_ID="${EXP_ID}_abn"
else
  if [[ ! -f $RESUME ]]; then
    echo "[E] Initial snapshot has not been set or was not found. Check RESUME='$RESUME'."
    exit 1
  fi

  RESUME_OPT="--resume $RESUME"

  if [[ ! -f $IS_WEIGHTS ]]; then
    echo "[W] Importance weights has not been set or were not found. Check IS_WEIGHTS='$IS_WEIGHTS'."
    echo "[W] Using uniform sampling"
  else
    EXTRA="DATASET.SAMPLE_WEIGHTS $IS_WEIGHTS $EXTRA"
  fi
fi

EXP_ID="${EXP_ID}_${SRC}_${NET}"
echo "RUN ID: $EXP_ID"

CURR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $CURR_DIR/utils.bash

CMD="python train.py --dataloader $DS --cfg $CFG --exp $EXP --run $EXP_ID $RESUME_OPT --set $EXTRA"
LOG_FILE=$LOG_DIR/${EXP_ID}.log

check_rundir $LOG_DIR $EXP_ID

export OMP_NUM_THREADS=12
export WORLD_SIZE=1

echo $CMD
echo "Logging to: $LOG_FILE"

nohup $CMD > $LOG_FILE 2>&1 &
sleep 1
tail -f $LOG_FILE
