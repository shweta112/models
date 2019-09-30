# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_kaist_tf_record.py \
        --image_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/images/set00/Visible_V000 \
        --annotations_dir=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/annotations-xml/set00/V000/ \
        --output_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/train.record-set00-v000 \
        --output_val_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/tf-records/val.record-set00-v000 \
        --val_split=0.1 \
        --label_map_path=/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/hd_label_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import logging
import io
import os

from lxml import etree
import cv2
import PIL.Image
import numpy as np
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags
flags.DEFINE_string('image_dir', '', 'Path to image directory.')
flags.DEFINE_string('add_image_dir', '', 'Path to additional channel')
flags.DEFINE_string('annotations_dir', '', 'Path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('output_val_path', '', 'Path to validation output TFRecord')
flags.DEFINE_float('val_split', None, 'Validation split')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                     'difficult instances')
FLAGS = flags.FLAGS

np.random.seed(10101)

def dict_to_tf_example(data, image_dir, add_image_dir, label_map_dict,
                       ignore_difficult_instances=False):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    image_dir: Path to image directory.
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  full_path = os.path.join(image_dir, data['filename'].split('/')[-1] + '.png')
  add_path = os.path.join(add_image_dir, data['filename'].split('/')[-1] + '.png')

  with tf.gfile.GFile(full_path, 'rb') as fid:
      encoded_png = fid.read() # bytes
  encoded_png_io = io.BytesIO(encoded_png) # bytes buffer
  image = PIL.Image.open(encoded_png_io)

  encoded_add_png = None
  if add_image_dir:
    with tf.gfile.GFile(add_path, 'rb') as fid:
      encoded_add_png = fid.read() # bytes

  '''
    # Read your image and extra inputs
    image = cv2.imread(full_path, cv2.IMREAD_ANYDEPTH)
    # Encode your input as string
    encoded_image = image.tostring()
  '''

  if image is None:
      raise ValueError('Image %s is not valid'%full_path)

  key = hashlib.sha256(encoded_png).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  occlusion = []


  if 'object' in data:
    for obj in data['object']:
      try:
          class_name = label_map_dict[obj['name']]
      except KeyError as e:
          print(full_path + ' has unknown object %s!'%obj['name'])
          continue

      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue

      difficult_obj.append(int(difficult))

      xmin.append(float(obj['bndbox']['xmin']) / width)
      ymin.append(float(obj['bndbox']['ymin']) / height)
      xmax.append(float(obj['bndbox']['xmax']) / width)
      ymax.append(float(obj['bndbox']['ymax']) / height)
      classes_text.append(obj['name'].encode('utf8'))
      classes.append(class_name)
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))
      occlusion.append(int(obj['occlusion']))


  if add_image_dir:
      example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': dataset_util.int64_feature(height),
          'image/width': dataset_util.int64_feature(width),
          'image/filename': dataset_util.bytes_feature(
              data['filename'].encode('utf8')),
          'image/source_id': dataset_util.bytes_feature(
              data['filename'].encode('utf8')),
          'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
          # 'image/channels': dataset_util.int64_feature(image.shape[-1]),
          'image/encoded': dataset_util.bytes_feature(encoded_png),
          'image/additional_channels/encoded': dataset_util.bytes_feature(encoded_add_png),
          'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
          'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
          'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
          'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
          'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
          'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
          'image/object/class/label': dataset_util.int64_list_feature(classes),
          'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
          'image/object/truncated': dataset_util.int64_list_feature(truncated),
          'image/object/view': dataset_util.bytes_list_feature(poses),
          'image/object/occlusion': dataset_util.int64_list_feature(occlusion),
      }))
      return example

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      # 'image/channels': dataset_util.int64_feature(image.shape[-1]),
      'image/encoded': dataset_util.bytes_feature(encoded_png),
      'image/format': dataset_util.bytes_feature('png'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
      'image/object/occlusion': dataset_util.int64_list_feature(occlusion),
  }))
  return example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    image_dir = FLAGS.image_dir
    add_image_dir = FLAGS.add_image_dir
    annotations_dir = FLAGS.annotations_dir
    logging.info('Reading from dataset: ' + annotations_dir)
    examples_list = os.listdir(annotations_dir)
    print('Total %d images'%len(examples_list))
    val_list = []
    val_writer = None

    if FLAGS.val_split:
        val_writer = tf.python_io.TFRecordWriter(FLAGS.output_val_path)
        np.random.shuffle(examples_list)
        val_start = int((1 - FLAGS.val_split)*len(examples_list))
        print('Validation start idx %d'%val_start)
        val_list = examples_list[val_start:]
        examples_list = examples_list[:val_start]


    for idx, example in enumerate(examples_list):
        if example.endswith('.xml'):
            if idx % 100 == 0:
                print('On image %d of %d' % (idx, len(examples_list)))

            path = os.path.join(annotations_dir, example)
            with tf.gfile.GFile(path, 'r') as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

            tf_example = dict_to_tf_example(data, image_dir, add_image_dir, label_map_dict)
            writer.write(tf_example.SerializeToString())

    writer.close()

    if val_writer:
        for idx, example in enumerate(val_list):
            if example.endswith('.xml'):
                if idx % 100 == 0:
                    print('On image %d of %d' % (idx, len(val_list)))

                path = os.path.join(annotations_dir, example)
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

                tf_example = dict_to_tf_example(data, image_dir, add_image_dir, label_map_dict)
                val_writer.write(tf_example.SerializeToString())

        val_writer.close()



if __name__ == '__main__':
  tf.app.run()
