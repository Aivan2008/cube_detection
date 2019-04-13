#!/usr/bin/python
# coding: utf-8
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
import os
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from object_detection_msgs.msg import DetectorResult as DRes
from object_detection_msgs.msg import DetectorBBox as DBox
from nav_msgs.msg import Odometry
import tf

#################################################
bridge = CvBridge()
rospack = rospkg.RosPack()
pub = rospy.Publisher('/samsung/BBoxes', DRes, queue_size=20)
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
#wheights_path = cur_dir+'/data/'+"sqd_1k_anchors_176eps.hdf5"
#CONFIG        = cur_dir+'/data/'+"squeeze.config"
wheights_path = cur_dir+'/data/'+"model.180-0.99.hdf5"
CONFIG        = cur_dir+'/data/'+"squeeze_cube_640x480.config"
#################################################
cfg = load_dict(CONFIG)
#Required global vars to hold initialized everything
#between detector runs
#workaround, may be wrong
squeeze=None
sgd = None
model = None
image_index=0
font = cv2.FONT_HERSHEY_SIMPLEX
#################################################
#   ___      __          __  _         
#  / _ \___ / /____ ____/ /_(_)__  ___ 
# / // / -_) __/ -_) __/ __/ / _ \/ _ \
#/____/\__/\__/\__/\__/\__/_/\___/_//_/                                      
#################################################
from multiprocessing import Lock
odom_mutex=Lock()
odometry_pose = []
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
def OdometryCallback(msg):
    global odometry_pose, odom_mutex
    with odom_mutex:
        orient = np.array([msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w])
        euler = tf.transformations.euler_from_quaternion(orient)
        odometry_pose = [msg.pose.pose.position.x, msg.pose.pose.position.y, euler[2]]
        
def Callback(data):
    global squeeze, sgd, model, image_index, previous_bboxes, odometry_pose, odom_mutex
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
    
    #print "Image received!"
    time_now = rospy.get_rostime()
    now = time_now.secs
    #Probably will be needed to reject too old images
    #Will be work properly only if computers syncronized well 
    #(connected to same wifi or smth like this)
    #diff = now - data.header.stamp.secs
    #print "Diff = ", diff
    #if diff > 1:
    #   print "Difference is too large"
    #   return
    before_pred = rospy.get_rostime()
    try:
        #This is for image_raw
        #cv2_img = bridge.imgmsg_to_cv2(data, "bgr8")

        #This is for image_raw/compressed
        np_arr = np.fromstring(data.data, np.uint8)
        cv2_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        noww = rospy.get_rostime()
        #print "now: {0}.{1}".format(noww.secs, noww.nsecs), "message: {0}.{1}".format(data.header.stamp.secs, data.header.stamp.nsecs)

        
        orig_h =cv2_img.shape[0]
        orig_w =cv2_img.shape[1]
        
    except CvBridgeError, e:
        print(e)
    else:
        img_ = cv2_img.astype(np.float32, copy=False)
        img_ = (img_ - np.mean(img_)) / np.std(img_)
        img_ = cv2.resize(img_, (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
        
        img_for_pred = np.expand_dims(img_, axis=0)
        #print (img_for_pred.shape)
        
        e1 = cv2.getTickCount()
        y_pred = model.predict(img_for_pred)
        e2 = cv2.getTickCount()
        tme = (e2 - e1) / cv2.getTickFrequency()
        #print("prediction time: {0}".format(tme))
        
        #print("  predicted something")


        all_filtered_boxes, all_filtered_classes, all_filtered_scores = filter_batch(y_pred, cfg)

        all_det_boxes = []
        #if not os.path.exists(cur_dir+"/images"):
            #os.makedirs(cur_dir+"/images")
        #if not os.path.exists(cur_dir+"/txt_files"):
            #os.makedirs(cur_dir+"/txt_files")
        #image_name = cur_dir+"/images/img_{0}.jpg".format(str(image_index).zfill(4))
        #txtfile_name = cur_dir+"/txt_files/img_{0}.txt".format(str(image_index).zfill(4))
        #cv2.imwrite(image_name, cv2_img)
        #txtfile = open(txtfile_name, 'w')
        image_index+=1
        for j, det_box in enumerate(all_filtered_boxes[0]):
            # transform into xmin, ymin, xmax, ymax
            det_box = bbox_transform_single_box(det_box)
            x_scale = float(orig_w) / cfg.IMAGE_WIDTH
            y_scale = float(orig_h) / cfg.IMAGE_HEIGHT
            det_box[0], det_box[2] = int(det_box[0] * x_scale), int(det_box[2] * x_scale)
            det_box[1], det_box[3] = int(det_box[1] * y_scale), int(det_box[3] * y_scale)

            
            
            
            #txtfile.write("{0} {1} {2} {3} {4}\n".format(det_box[0], det_box[1], det_box[2], det_box[3], all_filtered_scores[0][j]))
            # possibly will be needed to gen out image
            cv2.rectangle(cv2_img, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (255, 10, 10), 2)
            cv2.putText(cv2_img, cfg.CLASS_NAMES[all_filtered_classes[0][j]] + " " + str(all_filtered_scores[0][j]),
                        (det_box[0], det_box[1]), font, 1, (255, 10, 10), 2, cv2.LINE_AA)
            
            det_box.append(all_filtered_scores[0][j])
            #print(det_box)
            all_det_boxes.append(det_box)

        #cv2.imshow("img", cv2_img)
        #cv2.waitKey(1)

        search_zone_perc = rospy.get_param("/cube_detector/double_area_size", 1.2)
        search_zone = max(orig_w, orig_h)*search_zone_perc/2
        # Фильтрация детекций
        #Обычно самый маленький - самый классный
        #Берем обнаружение, считаем размер, считаем координаты
        near_threshold_r2 = 225 #Радиус 8 пкс, в квадрате 64
        positioned_boxes = []
        if len(all_det_boxes)>0:
            bbox = all_det_boxes[0]
            bbox_x = (bbox[0]+bbox[2])/2
            bbox_y = (bbox[1]+bbox[3])/2
            bbox_w = abs(bbox[0]-bbox[2])
            bbox_h = abs(bbox[1]-bbox[3])
            positioned_boxes.append([[bbox_x, bbox_y,bbox_w, bbox_h], bbox])
        for bbox in all_det_boxes:
            bbox_x = (bbox[0]+bbox[2])/2
            bbox_y = (bbox[1]+bbox[3])/2
            bbox_w = abs(bbox[0]-bbox[2])
            bbox_h = abs(bbox[1]-bbox[3])
            conf = bbox[4]
            replace_id=-1
            new_box = False
            for bb_id, current_bbox in enumerate(positioned_boxes):
                curr_x, curr_y, curr_w, curr_h = current_bbox[0]
                curr_conf = current_bbox[1][4]
                dx = abs(bbox_x-curr_x)
                dy = abs(curr_y-bbox_y)
                r2 = dx*dx + dy*dy
                #print r2
                #Мы совпали, можно выбирать
                if r2<near_threshold_r2:
                    new_box=False
                    if conf>curr_conf:
                        replace_id = bb_id
                        break
                else:
                    new_box=True

            if new_box:
                positioned_boxes.append([[bbox_x, bbox_y,bbox_w, bbox_h], bbox])
            else:
                if replace_id>=0:
                    positioned_boxes[replace_id]=[[bbox_x, bbox_y,bbox_w, bbox_h], bbox]
                                   
        filt_det_bboxes = [x[1] for x in positioned_boxes]
                
        #print "----", len(filt_det_bboxes), len(all_det_boxes), '-----'
        #p = []
        with odom_mutex:
            p = odometry_pose
        if len(filt_det_bboxes)>0:
            d_res = DRes()
            d_res.header.stamp = data.header.stamp
            if len(p)==3:
                d_res.x = p[0]
                d_res.y = p[1]
                d_res.angleZ = p[2]
            else:
                d_res.x = 0
                d_res.y = 0
                d_res.angleZ = 0
            for bbox in filt_det_bboxes:
                d_box = DBox()
                d_box.label = "cube"
                d_box.conf = bbox[4]
                d_box.x_min = bbox[0]
                d_box.y_min = bbox[1]
                d_box.x_max = bbox[2]
                d_box.y_max = bbox[3]
                d_res.res.append(d_box)
            #rospy.loginfo(d_res)
            pub.publish(d_res)
    after_pred = rospy.get_rostime()
    #print "Iteration took: {0}.{1}".format(after_pred.secs-before_pred.secs, after_pred.nsecs-before_pred.nsecs)
#########################################################################
#   ___  ____  ____  __  ___     _        ________                   __
#  / _ \/ __ \/ __/ /  |/  /__ _(_)__    /_  __/ /  _______ ___ ____/ /
# / , _/ /_/ /\ \  / /|_/ / _ `/ / _ \    / / / _ \/ __/ -_) _ `/ _  / 
#/_/|_|\____/___/ /_/  /_/\_,_/_/_//_/   /_/ /_//_/_/  \__/\_,_/\_,_/                                                                    
#########################################################################
def Detector(argv):
    #print "Start "
    rospy.init_node('detector')
    if not rospy.has_param('/cube_detector/trj_len'):
        rospy.set_param('/cube_detector/trj_len', 2)

    #Два размера куба от прошлого положения
    if not rospy.has_param('/cube_detector/double_area_size'):
        rospy.set_param('/cube_detector/double_area_size', 1.5)

    #Два размера куба от прошлого положения
    if not rospy.has_param('/cube_detector/search_area_size'):
        rospy.set_param('/cube_detector/search_area_size', 2.0)
    
    #if len(argv) < 2:
    #video_src = '/usb_cam_front/image_raw'
    video_src = '/usb_cam_front/image_raw/compressed'
    #print "Camera: ", video_src
    #else:
    #    	video_src = argv[1]OdometryCallback(msg):
    rospy.Subscriber(video_src, CompressedImage, Callback, queue_size=1, buff_size=2092800)
    rospy.Subscriber("/kursant_driver/odom", Odometry, OdometryCallback, queue_size=1, buff_size=2092800)
    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    #cv2.startWindowThread()

    rospy.spin()
    
    #cv2.destroyWindow('img')
###################################
#   __  ___     _    
#  /  |/  /__ _(_)__ 
# / /|_/ / _ `/ / _ \
#/_/  /_/\_,_/_/_//_/
###################################
if __name__ == '__main__':
	print "Cubes detector started"
	Detector(sys.argv)

exit(0)
