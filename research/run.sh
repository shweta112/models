#!/usr/bin/env bash


# Use this script to run the training for any of the object detection models
# Apart from PIPELINE_CONFIG_PATH and MODEL_DIR, set the NUM_TRAIN_STEPS to desired value before running the script.
# Refer to model_main.py for detailed information about available arguments

## From the tensorflow/models/research/ directory
#PIPELINE_CONFIG_PATH=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/ssd_inception_v2_hd.config
#MODEL_DIR=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_inception_v2_coco_2018_01_28/rgb
#NUM_TRAIN_STEPS=50000
#SAMPLE_1_OF_N_EVAL_EXAMPLES=1
#python object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --num_train_steps=${NUM_TRAIN_STEPS} \
#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
#    --alsologtostderr

## From the tensorflow/models/research/ directory
#PIPELINE_CONFIG_PATH=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v2_hd.config
#MODEL_DIR=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_mobilenet_v2_coco_2018_03_29/rgbt
#SAMPLE_1_OF_N_EVAL_EXAMPLES=1
#python object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
#    --alsologtostderr

## From the tensorflow/models/research/ directory
#PIPELINE_CONFIG_PATH=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v2_hd_.config
#MODEL_DIR=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_mobilenet_v2_coco_2018_03_29/rgbt_san
#SAMPLE_1_OF_N_EVAL_EXAMPLES=1
#python object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
#    --alsologtostderr


##### Faster R-CNN fusion techniques

# From the tensorflow/models/research/ directory
#PIPELINE_CONFIG_PATH=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/faster_rcnn_resnet101_hd_.config
#MODEL_DIR=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/added/
#SAMPLE_1_OF_N_EVAL_EXAMPLES=1
#python object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
#    --alsologtostderr

# From the tensorflow/models/research/ directory
#PIPELINE_CONFIG_PATH=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/faster_rcnn_resnet101_hd_f.config
#MODEL_DIR=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/fused/
#SAMPLE_1_OF_N_EVAL_EXAMPLES=1
#python object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
#    --alsologtostderr


####### QUANTISED TPU COMPATIBLE #######
## From the tensorflow/models/research/ directory
#PIPELINE_CONFIG_PATH=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/configs/pipeline_mobilenet_v2_ssd_retrain_whole_model_hd.config
#MODEL_DIR=/media/shweta.mahajan/Daten/Coral_Edge_TPU/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/rgbt
#SAMPLE_1_OF_N_EVAL_EXAMPLES=1
#python object_detection/model_main.py \
#    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#    --model_dir=${MODEL_DIR} \
#    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
#    --alsologtostderr

# From the tensorflow/models/research/ directory
PIPELINE_CONFIG_PATH=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/configs/pipeline_mobilenet_v2_ssd_retrain_whole_model_hd.config
MODEL_DIR=/media/shweta.mahajan/Daten/Coral_Edge_TPU/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/ft_fraunhofer
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

