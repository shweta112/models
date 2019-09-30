#!/usr/bin/env bash
python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set06/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-000-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set06/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-001-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/Visible_V002 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/LWIR_V002 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set06/V002/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-002-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/Visible_V003 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/LWIR_V003 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set06/V003/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-003-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/Visible_V004 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/LWIR_V004 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set06/V004/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-004-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set07/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set07/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set07/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-005-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set07/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set07/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set07/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-006-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set07/Visible_V002 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set07/LWIR_V002 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set07/V002/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-007-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set08/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set08/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set08/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-008-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set08/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set08/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set08/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-009-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set08/Visible_V002 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set08/LWIR_V002 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set08/V002/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-010-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set09/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set09/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set09/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-011-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set10/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set10/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set10/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-012-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set10/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set10/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set10/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-013-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set11/Visible_V000 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set11/LWIR_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set11/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-014-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set11/Visible_V001 \
        --add_image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set11/LWIR_V001 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set11/V001/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test-rgbt.record-015-of-016 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt

