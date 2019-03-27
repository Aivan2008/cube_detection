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
bridge = CvBridge()
rospack = rospkg.RosPack()
pub = rospy.Publisher('BBoxes', DRes, queue_size=20)
cur_dir = rospack.get_path('cube_detection')
#print cur_dir
#image_pub = rospy.Publisher('Image_result', Image)
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
#   ___                          __            
#  / _ \___ ________ ___ _  ___ / /____ _______
# / ___/ _ `/ __/ _ `/  ' \/ -_) __/ -_) __(_-<
#/_/   \_,_/_/  \_,_/_/_/_/\__/\__/\__/_/ /___/
#################################################
wheights_path = cur_dir+'/data/'+"sqd_1k_anchors_176eps.hdf5"
CONFIG        = cur_dir+'/data/'+"squeeze.config"
#################################################
cfg = load_dict(CONFIG)

squeeze=None
sgd = None
model = None
#################################################
#   ___      __          __  _         
#  / _ \___ / /____ ____/ /_(_)__  ___ 
# / // / -_) __/ -_) __/ __/ / _ \/ _ \
#/____/\__/\__/\__/\__/\__/_/\___/_//_/                                      
#################################################

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

def Callback(data):
    global squeeze, sgd, model
    if squeeze==None:
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
    
    print "Image received!"
    time_now = rospy.get_rostime()
    now = time_now.secs
    #diff = now - data.header.stamp.secs
    #print "Diff = ", diff
    #if diff > 1:
    #   print "Difference is too large"
    #   return
	#print "Before try!"
    try:
        cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")
        orig_h =cv2_img.shape[0]
        orig_w =cv2_img.shape[1]
        cv2.imshow("img", cv2_img)
        cv2.waitKey(1)
    except CvBridgeError, e:
        print(e)
    else:
        #img_name = "sm1.jpg"
        #print img_name
        # open img
        #img_ = cv2.imread(img_name).astype(np.float32, copy=False)
        img_ = cv2_img.astype(np.float32, copy=False)
        img_ = (img_ - np.mean(img_)) / np.std(img_)
        img_ = cv2.resize(img_, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
        img_for_pred = np.expand_dims(img_, axis=0)
        print (img_for_pred.shape)

        y_pred = model.predict(img_for_pred)
        print("  predicted something")


        all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, cfg)

        all_det_boxes = []
        for j, det_box in enumerate(all_filtered_boxes[0]):
            # transform into xmin, ymin, xmax, ymax
            det_box = bbox_transform_single_box(det_box)
            x_scale = float(orig_w) / cfg.IMAGE_WIDTH
            y_scale = float(orig_h) / cfg.IMAGE_HEIGHT
            det_box[0], det_box[2] = int(det_box[0] * x_scale), int(det_box[2] * x_scale)
            det_box[1], det_box[3] = int(det_box[1] * y_scale), int(det_box[3] * y_scale)


            # add rectangle and text
            #cv2.rectangle(img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (255, 10, 10), 2)
            #cv2.putText(img, cfg.CLASS_NAMES[all_filtered_classes[i][j]] + " " + str(all_filtered_scores[i][j]),
            #            (det_box[0], det_box[1]), font, 1, (255, 10, 10), 2, cv2.LINE_AA)
            det_box.append(all_filtered_scores[0][j])
            print(det_box)
            all_det_boxes.append(det_box)

        
        if len(all_det_boxes)>0:
            d_res = DRes()
            d_res.header.stamp = data.header.stamp
            for bbox in all_det_boxes:
                d_box = DBox()
                d_box.label = "cube"
                d_box.conf = bbox[4]
                d_box.x_min = bbox[0]
                d_box.y_min = bbox[1]
                d_box.x_max = bbox[2]
                d_box.y_max = bbox[3]
                d_res.res.append(d_box)
            rospy.loginfo(d_res)
            pub.publish(d_res)


	#global graph
	#img_full = cv2_img.copy()
	
	#img_result = cv2.resize(img_full, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
'''
	with graph.as_default():
	    	result_detect =  img_test.run(cv2_img, False)
	    	print result_detect
	    	if not result_detect:
	    	    	rospy.loginfo('Object not found')
	    	    	return
	    	if not result_detect[0]:
	    	    	rospy.loginfo('Result is empty')
	    	    	return
	    	global pub
	    	global image_pub

	    	height, width, channels = img_result.shape
	
	    	d_out = DOut()
	    	d_out.labels = result_detect[0]
	    	d_out.conf = result_detect[1]
	    	d_out.x_min = result_detect[2]*width
	    	d_out.y_min = result_detect[3]*height
	    	d_out.x_max = result_detect[4]*width
	    	d_out.y_max = result_detect[5]*height
	    	d_out.header.stamp = data.header.stamp

	    	index = 0
	    	for label in d_out.labels:
	    	    	x1 = int(d_out.x_min[index])
	    	    	x2 = int(d_out.x_max[index])
	    	    	y1 = int(d_out.y_min[index])
	    	    	y2 = int(d_out.y_max[index])

	    	    	cv2.rectangle(img_result, (x1, y1), (x2, y2), (0,0,255), 2)
	    	    	index+=1
	    	    	

	    	
	    	#cv2.rectangle(img_result, (0, 0), (100, 100), (0,0,255), 1)
	    	image_pub.publish(bridge.cv2_to_imgmsg(img_result, "bgr8"))
	    	rospy.loginfo(d_out)
	    	pub.publish(d_out)
            '''

def Detector(argv):
	#print "Start "
	rospy.init_node('detector')
	#if len(argv) < 2:
	video_src = '/usb_cam_front/image_raw'
	#print "Camera: ", video_src
	#else:
	#    	video_src = argv[1]

	
	rospy.Subscriber(video_src, Image, Callback, queue_size=1, buff_size=2092800)
	rospy.spin()

if __name__ == '__main__':
	print "Main"
	Detector(sys.argv)

exit(0)
#################################################
'''
  ____  __   __  __               __ 
 / __ \/ /__/ / / /________ ____ / / 
/ /_/ / / _  / / __/ __/ _ `(_-</ _ \
\____/_/\_,_/  \__/_/  \_,_/___/_//_/
                                     
'''
#################################################

'''
img_name = "sm1.jpg"

CONFIG = "squeeze.config"

#wheights_path = 'kitti.hdf5'
#wheights_path = "log/checkpoints/kitti.hdf5"
#wheights_path= "squeezedet_start_from_kitti_50eps.hdf5"
#wheights_path ="model.10-11.06.hdf5"
#wheights_path ="sqd_300eps_correct_ann.hdf5"
wheights_path ="sqd_1k_anchors_176eps.hdf5"
#wheights_path = "sqd_400eps_lr01_start_from_kitti.hdf5"
#wheights_path = "sqd_350eps.hdf5"



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

# create config object
cfg = load_dict(CONFIG)



# open img
img = cv2.imread(img_name).astype(np.float32, copy=False)
orig_h =img.shape[0]
orig_w =img.shape[1]
# subtract means
img = (img - np.mean(img)) / np.std(img)
img = cv2.resize(img, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
img_for_pred = np.expand_dims(img, axis=0)
print (img_for_pred.shape)


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



y_pred = model.predict(img_for_pred)
print("  predicted something")


all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, cfg)
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.imread(img_name)

for j, det_box in enumerate(all_filtered_boxes[0]):
    print det_box
    ''
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
    ''

cv2.imwrite("sm1_r.jpg", img)
'''
