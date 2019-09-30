#!/usr/bin/env bash
TF_RECORD_FILES=$(ls /media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test.record-000-of-016 | tr '\n' ',')

python object_detection/inference/infer_detections_txt.py \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_path=/media/shweta.mahajan/Daten/GitHub/mAP/input/detection-results \
  --inference_graph=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_inception_v2_coco_2018_01_28/rgb/frozen_inference_graph.pb

TF_RECORD_FILES=$(ls /media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/test.record-00[2-3]-of-016 | tr '\n' ',')

python object_detection/inference/infer_detections_txt.py \
  --input_tfrecord_paths=$TF_RECORD_FILES \
  --output_path=/media/shweta.mahajan/Daten/GitHub/mAP/input/detection-results \
  --inference_graph=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/ssd_inception_v2_coco_2018_01_28/rgb/frozen_inference_graph.pb
