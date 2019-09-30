#!/usr/bin/env bash

cd /usr/local/lib/python3.5/dist-packages/edgetpu/demo

python3 object_detection.py \
--model /media/shweta.mahajan/Daten/Coral_Edge_TPU/ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/rgbt/tflite/output_tflite_graph.tflite \
--label /media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt \
--input /media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set06/V000/I00580.png