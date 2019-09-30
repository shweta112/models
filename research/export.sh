#!/usr/bin/env bash

# Use this script to export checkpoint to frozen graph for inference
# If the image tensor is not (None, None, None, 3) specify the shape using the input_shape argument
#   input_shape: Only integers are allowed. Use -1 for unknown shape (None)
# Refer to export_inference_graph.py for detailed information about available arguments

#python object_detection/export_inference_graph.py \
#    --input_type=image_tensor \
#    --pipeline_config_path=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/ssd_inception_v2_hd.config \
#    --trained_checkpoint_prefix=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_inception_v2_coco_2018_01_28/rgb/model.ckpt-50000 \
#    --output_directory=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_inception_v2_coco_2018_01_28/rgb

python object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --input_shape=-1,-1,-1,4 \
    --pipeline_config_path=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/ssd_mobilenet_v2_hd.config \
    --trained_checkpoint_prefix=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_mobilenet_v2_coco_2018_03_29/rgbt/model.ckpt-292294 \
    --output_directory=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_mobilenet_v2_coco_2018_03_29/rgbt

#python object_detection/export_inference_graph.py \
#    --input_type=image_tensor \
#    --input_shape=-1,-1,-1,4 \
#    --pipeline_config_path=/media/shweta.mahajan/Daten/GitHub/tensorflow/models/research/object_detection/samples/configs/faster_rcnn_resnet101_hd_.config \
#    --trained_checkpoint_prefix=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/rgbt/model.ckpt-359612 \
#    --output_directory=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/rgbt

