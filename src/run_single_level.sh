#!/bin/bash

# Runs scan completion with a single level over a set of scenes.

# Parameter section begins here. Edit to change number of test scenes, which model to use, output path.
MAX_NUM_TEST_SCENES=1
NUM_HIERARCHY_LEVELS=1
BASE_OUTPUT_DIR=/home/pganti/git/ScanComplete/vis

# Fill in path to test scenes
TEST_SCENES_PATH_=/home/pganti/git/ScanComplete/data/Placenote/vox19

# Fill in model to use here
PREDICT_SEMANTICS=0
MODEL='/home/pganti/git/ScanComplete/models/completion/hierarchy3of3'
MODEL_CHECKPOINT='model.ckpt'

# Specify output folders for each hierarchy level.
OUTPUT_FOLDER=${BASE_OUTPUT_DIR}/Placenote

# End parameter section.


# Run hierarchy.

# ------- hierarchy level 3 ------- #

IS_BASE_LEVEL=1
HIERARCHY_LEVEL=3

# vox19 - HEIGHT_INPUT=16
# vox9 - HEIGHT_INPUT=32
# vox5 - HEIGHT_INPUT=64
HEIGHT_INPUT=16

# Go through all test scenes.
count=1
for scene in $TEST_SCENES_PATH/*__0__.tfrecords; do
  echo "Processing hierarchy level 3, scene $count of $MAX_NUM_TEST_SCENES: $scene".
  python complete_scan.py \
    --alsologtostderr \
    --base_dir="${MODEL}" \
    --model_checkpoint="${MODEL_CHECKPOINT}" \
    --height_input="${HEIGHT_INPUT}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER}" \
    --input_scene="${scene}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done
