import os
import sys
import numpy as np
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

# For Downloading
import six.moves.urllib as urllib
import tarfile
import zipfile

import cv2

# Load the input videos
videoCapture = cv2.VideoCapture('test_images/test.mp4')

fps = videoCapture.get(cv2.CAP_PROP_FPS)
v_width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
v_height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (v_width, v_height)
numFrames = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

# Video format, I420-avi, MJPG-mp4
videoWriter = cv2.VideoWriter('test_images/result.mp4', cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

# What model to load.
MODEL_NAME = 'ssd_inception_v2_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def detect(image_np, detectiongraph, session, Classes, Counter):
	# Definite input and output Tensors for detection_graph
    image_tensor = detectiongraph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detectiongraph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detectiongraph.get_tensor_by_name('detection_scores:0')
    detection_classes = detectiongraph.get_tensor_by_name('detection_classes:0')
    num_detections = detectiongraph.get_tensor_by_name('num_detections:0')

    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = session.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Store the class(es) of each frame
    for i in range(NUM_CLASSES):
    	if np.squeeze(scores)[i] > 0.5 and np.squeeze(classes)[i] in category_index.keys():
    		Classes[Counter].append(category_index[np.squeeze(classes)[i]]['name'])
    	else:
    		break
      
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    return image_np

# Create a 2-d list storing classes
classes = [[] for i in range(numFrames)]

# Multicore and multi-thread parallel computing configuration
config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage  
            inter_op_parallelism_threads = 8,
            intra_op_parallelism_threads = 4,
            log_device_placement=True)

with detection_graph.as_default():
	with tf.Session(graph=detection_graph, config = config) as sess:
  		success, frame = videoCapture.read()
  		counter = 0
  		while success:
  			frame = detect(frame, detection_graph, sess, classes, counter)
  			cv2.putText(frame,str(counter/fps)+"s",(20,20),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),thickness = 2)
  			cv2.imshow('frame', frame)
  			cv2.waitKey(1000/int(fps))
  			videoWriter.write(frame)
  			success, frame = videoCapture.read()
  			counter += 1

# Save classes to a file
np.save("test_images/classes.npy", classes)

videoCapture.release()
cv2.destroyAllWindows()