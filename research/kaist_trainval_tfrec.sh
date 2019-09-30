#!/usr/bin/env bash

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV000 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V000/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-000-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-000-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV001 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V001/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-001-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-001-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV002 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V002/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-002-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-002-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV003 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V003/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-003-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-003-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV004 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V004/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-004-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-004-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV005 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V005/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-005-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-005-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV006 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V006/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-006-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-006-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV007 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V007/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-007-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-007-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set00/aV008 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V008/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-008-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-008-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set01/aV000 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V000/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-009-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-009-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set01/aV001 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V001/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-010-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-010-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set01/aV002 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V002/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-011-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-011-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set01/aV003 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V003/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-012-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-012-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set01/aV004 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V004/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-013-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-013-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set01/aV005 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V005/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-014-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-014-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set02/aV000 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V000/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-015-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-015-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set02/aV001 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V001/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-016-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-016-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set02/aV002 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V002/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-017-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-017-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set02/aV003 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V003/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-018-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-018-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set02/aV004 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V004/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-019-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-019-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set03/aV000 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set03/V000/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-020-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-020-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set03/aV001 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set03/V001/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-021-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-021-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set04/aV000 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set04/V000/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-022-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-022-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set04/aV001 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set04/V001/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-023-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-023-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/images/set05/aV000 \
        --annotations_dir=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set05/V000/ \
        --output_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-a.record-024-of-025 \
        --output_val_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-a.record-024-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Transcend2TB/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt
