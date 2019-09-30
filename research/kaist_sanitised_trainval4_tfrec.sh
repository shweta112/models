#!/usr/bin/env bash
python object_detection/dataset_tools/create_sanitised_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/ \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/ \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/sanitized_annotations/sanitized_annotations/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-san-rgbt.record \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-san-rgbt.record \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

