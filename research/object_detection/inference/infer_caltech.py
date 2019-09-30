"""
    Author: Shweta Mahajan

    File to infer detections and save them in text files and save images with bounding boxes (optional).
"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import time
import glob
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2

# This is needed since the notebook is stored in the object_detection folder.
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


tf.flags.DEFINE_boolean('discard_image_pixels', True,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')

FLAGS = tf.flags.FLAGS


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = '/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt/faster_rcnn_resnet101_lowproposals_coco_2018_01_28/rgb/frozen_inference_graph.pb'

PATH_TO_LABELS = os.path.join('/media/shweta.mahajan/Daten/GitHub/rgbt-ped-detection/data/kaist-rgbt', 'hd_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/media/shweta.mahajan/Daten/Human_Detection/Datasets/CaltechPedestrians/voc/images'


OUTPUT_PATH = '/media/shweta.mahajan/Daten/Human_Detection/Datasets/CaltechPedestrians/faster-rcnn_resnet101_50p'

files = sorted(os.listdir('/media/shweta.mahajan/Daten/Human_Detection/Datasets/CaltechPedestrians/Caltech_new_annotations/anno_test_1xnew'))
# print(len(files))

TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, file.split('jpg')[0] + 'png')
                     for file in files ]

print('Found %d images'%len(TEST_IMAGE_PATHS))

# Size, in inches, of the output images.
IMAGE_SIZE = (640, 512)
# IMAGE_SIZE = (640, 480)
CONF_THRESH = 0.5



def run_inference_for_single_image(image, graph):
  tic = time.time()
  # Run inference
  output_dict = sess.run(tensor_dict,
                         feed_dict={image_tensor: image})
  toc = (time.time() - tic)
  times.append(toc)
  print('Inference time %f'%toc)

  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  if 'detection_masks' in output_dict:
    output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


times = []
with detection_graph.as_default():
    with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, IMAGE_SIZE[1], IMAGE_SIZE[0])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
        # print([n.name for n in tf.get_default_graph().as_graph_def().node])
        # return

        for image_path in TEST_IMAGE_PATHS:
          # print(image_path)
          name = image_path.split('/')[-1]

          # image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
          # image_np = load_image_into_numpy_array(image)
          image_np = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
          org_shape = image_np.shape
          # print(org_shape)
          cv2.copyMakeBorder(image_np, 0, IMAGE_SIZE[1] - image_np.shape[0], 0, 0, cv2.BORDER_CONSTANT, None, 0)
          # print(image_np.shape)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
          image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
          output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

          num_det = output_dict['num_detections']
          detected_classes = output_dict['detection_classes']
          detected_scores = output_dict['detection_scores']
          detected_boxes = output_dict['detection_boxes']
          detections_string = ''

          # print(detected_boxes.T[1])
          for box in range(min(20, num_det)):
              # Ignore any predictions with confidence score lower than threshold
              if detected_scores[box] < CONF_THRESH:
                  continue
              # <class_name> <score> <left> <top> <right> <bottom>
              detections_string += '{} {} {} {} {} {}\n'.format(category_index[detected_classes[box]]['name'],
                                                                detected_scores[box],
                                                                detected_boxes[box][1] * image_np.shape[1],
                                                                detected_boxes[box][0] * image_np.shape[0],
                                                                detected_boxes[box][3] * image_np.shape[1],
                                                                detected_boxes[box][2] * image_np.shape[0])
          f_name = os.path.join(OUTPUT_PATH, name.split('.')[0] + '.jpg.txt')
          with open(f_name, 'w') as output_f:
              output_f.write(detections_string)

          img_vis = image_np[:org_shape[0], :org_shape[1], :3]
          # print(img_vis.shape)
          if not FLAGS.discard_image_pixels:
              # Visualization of the results of a detection.
              vis_util.visualize_boxes_and_labels_on_image_array(
                  img_vis,
                  output_dict['detection_boxes'],
                  output_dict['detection_classes'],
                  output_dict['detection_scores'],
                  category_index,
                  instance_masks=output_dict.get('detection_masks'),
                  use_normalized_coordinates=True,
                  line_thickness=3)
              # plt.figure(figsize=IMAGE_SIZE)
              # plt.imsave(os.path.join(OUTPUT_PATH, 'images', image_path.split('/')[-1]),
                         # img_vis)
              # plt.close()
              cv2.imwrite(os.path.join(OUTPUT_PATH, 'images', name), img_vis)

        print("Avg. inference time %f"%(sum(times)/len(times)))


