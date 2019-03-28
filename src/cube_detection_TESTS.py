#!/usr/bin/python
#################################################
#   ___  ____  ____
#  / _ \/ __ \/ __/
# / , _/ /_/ /\ \  
#/_/|_|\____/___/      
#################################################
import rospy
import sys
import getopt
import rospkg
import pickle
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from object_detection_msgs.msg import DetectorResult as DRes
from object_detection_msgs.msg import DetectorBBox as DBox
#################################################
#   ___      __          __          
#  / _ \___ / /____ ____/ /____  ____
# / // / -_) __/ -_) __/ __/ _ \/ __/
#/____/\__/\__/\__/\__/\__/\___/_/                                       
#################################################
from main.model.squeezeDet import SqueezeDet
from main.model.dataGenerator import generator_from_data_path, visualization_generator_from_data_path
import keras.backend as K
from keras import optimizers
import numpy as np
import argparse
from keras.utils import multi_gpu_model
from main.config.create_config import load_dict
import cv2
from main.model.evaluation import filter_batch
#################################################
bridge = CvBridge()
rospack = rospkg.RosPack()
pub = rospy.Publisher('BBoxes', DRes, queue_size=20)
cur_dir = rospack.get_path('cube_detection')
#print cur_dir
#image_pub = rospy.Publisher('Image_result', Image)
#################################################
#   ___                          __            
#  / _ \___ ________ ___ _  ___ / /____ _______
# / ___/ _ `/ __/ _ `/  ' \/ -_) __/ -_) __(_-<
#/_/   \_,_/_/  \_,_/_/_/_/\__/\__/\__/_/ /___/
#################################################
wheights_path = cur_dir+'/data/'+"sqd_1k_anchors_176eps.hdf5"
CONFIG        = cur_dir+'/data/'+"squeeze.config"
#################################################

#wheights_path = 'kitti.hdf5'
#wheights_path = "log/checkpoints/kitti.hdf5"
#wheights_path= "squeezedet_start_from_kitti_50eps.hdf5"
#wheights_path ="model.10-11.06.hdf5"
#wheights_path ="sqd_300eps_correct_ann.hdf5"
#wheights_path ="sqd_1k_anchors_176eps.hdf5"
#wheights_path = "sqd_400eps_lr01_start_from_kitti.hdf5"
#wheights_path = "sqd_350eps.hdf5"
# create config object
cfg = load_dict(CONFIG)
squeeze = SqueezeDet(cfg)
# dummy optimizer for compilation
sgd = optimizers.SGD(lr=cfg.LEARNING_RATE, decay=0, momentum=cfg.MOMENTUM,
                     nesterov=False, clipnorm=cfg.MAX_GRAD_NORM)
squeeze.model.compile(optimizer=sgd,
                              loss=[squeeze.loss], metrics=[squeeze.bbox_loss, squeeze.class_loss,
                                                            squeeze.conf_loss, squeeze.loss_without_regularization])
model = squeeze.model
i = 0
squeeze.model.load_weights(wheights_path)




def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

def bbox_transform_single_box(bbox):
    """convert a bbox of form [cx, cy, w, h] to [xmin, ymin, xmax, ymax]. Works
    for numpy array or list of tensors.
    """
    cx, cy, w, h = bbox
    out_box = [[]]*4
    out_box[0] = int(np.floor(cx-w/2))
    out_box[1] = int(np.floor(cy-h/2))
    out_box[2] = int(np.floor(cx+w/2))
    out_box[3] = int(np.floor(cy+h/2))

    return out_box




img_name = "sm1.jpg"
# open img
img = cv2.imread(img_name).astype(np.float32, copy=False)
orig_h =img.shape[0]
orig_w =img.shape[1]
# subtract means
img = (img - np.mean(img)) / np.std(img)
img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
img_for_pred = np.expand_dims(img, axis=0)
print (img_for_pred.shape)
y_pred = model.predict(img_for_pred)
print("  predicted something")


all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, cfg)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread(img_name)

for j, det_box in enumerate(all_filtered_boxes[0]):
    print det_box
    """
    # transform into xmin, ymin, xmax, ymax
    det_box = bbox_transform_single_box(det_box)
    x_scale = orig_w / cfg.IMAGE_WIDTH
    y_scale = orig_h / cfg.IMAGE_HEIGHT
    det_box[0], det_box[2] = int(det_box[0] * x_scale), int(det_box[2] * x_scale)
    det_box[1], det_box[3] = int(det_box[1] * y_scale), int(det_box[3] * y_scale)


    # add rectangle and text
    cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (255, 10, 10), 2)
    cv2.putText(img, cfg.CLASS_NAMES[all_filtered_classes[i][j]] + " " + str(all_filtered_scores[i][j]),
                (det_box[0], det_box[1]), font, 1, (255, 10, 10), 2, cv2.LINE_AA)
    print(det_box)
    """

cv2.imwrite("sm1_r.jpg", img)
'''
