#!/usr/bin/env bash

python object_detection/dataset_tools/create_fraunhofer_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/fraunhofer_dataset/train/vis_reg/cropped/ \
        --add_image_dir=/media/shweta.mahajan/Transcend2TB/fraunhofer_dataset/train/lwir/cropped/ \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/fraunhofer_dataset/train/annotations/cropped/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/fraunhofer_dataset/tf-records/train.record \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/fraunhofer_dataset/tf-records/val.record \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/fraunhofer_dataset/hd_label_map.pbtxt


