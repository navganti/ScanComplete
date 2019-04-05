#!/bin/bash

# Runs scan completion with hierarchical model over a set of scenes.

# Parameter section begins here. Edit to change number of test scenes, which model to use, output path.
MAX_NUM_TEST_SCENES=1
NUM_HIERARCHY_LEVELS=3
BASE_OUTPUT_DIR=/home/pganti/git/ScanComplete/vis

# Fill in path to test scenes
TEST_SCENES_PATH_3=/home/pganti/git/ScanComplete/data/testdata/vox19
TEST_SCENES_PATH_2=/home/pganti/git/ScanComplete/data/testdata/vox9
TEST_SCENES_PATH_1=/home/pganti/git/ScanComplete/data/testdata/vox5

# Fill in model to use here
PREDICT_SEMANTICS=0
HIERARCHY_LEVEL_3_MODEL='/home/pganti/git/ScanComplete/models/completion/hierarchy3of3'
HIERARCHY_LEVEL_2_MODEL='/home/pganti/git/ScanComplete/models/completion/hierarchy2of3'
HIERARCHY_LEVEL_1_MODEL='/home/pganti/git/ScanComplete/models/completion/hierarchy1of3'
MODEL_CHECKPOINT='model.ckpt'

# Specify output folders for each hierarchy level.
OUTPUT_FOLDER_3=${BASE_OUTPUT_DIR}/vis_level3
OUTPUT_FOLDER_2=${BASE_OUTPUT_DIR}/vis_level2
OUTPUT_FOLDER_1=${BASE_OUTPUT_DIR}/vis_level1

# End parameter section.


# Run hierarchy.

# ------- hierarchy level 1 ------- #

IS_BASE_LEVEL=1
HIERARCHY_LEVEL=1
HEIGHT_INPUT=16

# Go through all test scenes.
count=1
for scene in $TEST_SCENES_PATH_3/*__0__.tfrecords; do
  echo "Processing hierarchy level 3, scene $count of $MAX_NUM_TEST_SCENES: $scene".
  python complete_scan.py \
    --alsologtostderr \
    --base_dir="${HIERARCHY_LEVEL_3_MODEL}" \
    --model_checkpoint="${MODEL_CHECKPOINT}" \
    --height_input="${HEIGHT_INPUT}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER_3}" \
    --input_scene="${scene}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done

# ------- hierarchy level 2 ------- #

IS_BASE_LEVEL=0
HIERARCHY_LEVEL=2
HEIGHT_INPUT=32

# go thru all test scenes
count=1
for scene in $TEST_SCENES_PATH_2/*__0__.tfrecords; do
  echo "Processing hierarchy level 2, scene $count of $MAX_NUM_TEST_SCENES: $scene".
  python complete_scan.py \
    --alsologtostderr \
    --base_dir="${HIERARCHY_LEVEL_2_MODEL}" \
    --model_checkpoint="${MODEL_CHECKPOINT}" \
    --output_dir_prev="${OUTPUT_FOLDER_3}" \
    --height_input="${HEIGHT_INPUT}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER_2}" \
    --input_scene="${scene}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done

# ------- hierarchy level 3 ------- #

IS_BASE_LEVEL=0
HIERARCHY_LEVEL=3
HEIGHT_INPUT=64

# go thru all test scenes
count=1
for scene in $TEST_SCENES_PATH_1/*__0__.tfrecords; do
  echo "Processing hierarchy level 1, scene $count of $MAX_NUM_TEST_SCENES: $scene".
  python complete_scan.py \
    --alsologtostderr \
    --base_dir="${HIERARCHY_LEVEL_1_MODEL}" \
    --model_checkpoint="${MODEL_CHECKPOINT}" \
    --output_dir_prev="${OUTPUT_FOLDER_2}" \
    --height_input="${HEIGHT_INPUT}" \
    --hierarchy_level="${HIERARCHY_LEVEL}" \
    --num_total_hierarchy_levels="${NUM_HIERARCHY_LEVELS}" \
    --is_base_level="${IS_BASE_LEVEL}" \
    --predict_semantics="${PREDICT_SEMANTICS}" \
    --output_folder="${OUTPUT_FOLDER_1}" \
    --input_scene="${scene}"
  ((count++))
  if (( count > MAX_NUM_TEST_SCENES )); then
    break
  fi
done


