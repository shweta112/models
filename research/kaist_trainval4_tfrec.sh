#!/usr/bin/env bash
python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-000-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-000-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-001-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-001-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V002 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V002 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V002/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-002-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-002-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V003 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V003 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V003/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-003-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-003-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V004 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V004 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V004/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-004-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-004-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V005 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V005 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V005/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-005-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-005-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V006 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V006 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V006/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-006-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-006-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V007 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V007 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V007/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-007-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-007-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V008 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/LWIR_V008 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V008/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-008-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-008-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-009-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-009-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-010-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-010-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/Visible_V002 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/LWIR_V002 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V002/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-011-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-011-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/Visible_V003 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/LWIR_V003 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V003/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-012-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-012-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/Visible_V004 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/LWIR_V004 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V004/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-013-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-013-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/Visible_V005 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set01/LWIR_V005 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set01/V005/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-014-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-014-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-015-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-015-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-016-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-016-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/Visible_V002 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/LWIR_V002 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V002/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-017-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-017-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/Visible_V003 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/LWIR_V003 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V003/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-018-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-018-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/Visible_V004 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set02/LWIR_V004 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set02/V004/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-019-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-019-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set03/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set03/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set03/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-020-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-020-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set03/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set03/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set03/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-021-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-021-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set04/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set04/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set04/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-022-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-022-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set04/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set04/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set04/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-023-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-023-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set05/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set05/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set05/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train-rgbt.record-024-of-025 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val-rgbt.record-024-of-025 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt
