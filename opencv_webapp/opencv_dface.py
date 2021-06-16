from django.conf import settings
from django.core.files.storage import FileSystemStorage
import numpy as np
import cv2
import tensorflow as tf


def opencv_dface(cl, ps):



    ############################################################   DEFINE MODEL ###################################################################

    BODY_PARTS = {"nose": 0, "leftEye": 1, "rightEye": 2, "leftEar": 3, "rightEar": 4,
                    "leftShoulder": 5, "rightShoulder": 6, "leftElbow": 7, "rightElbow": 8, "leftWrist": 9,
                    "rightWrist": 10, "leftHip": 11, "rightHip": 12, "leftKnee": 13, "rightKnee": 14,
                    "leftAnkle": 15, "rightAnkle": 16}

    




    BODY_PARTS_COLORS = {"nose": (153,0,0), "leftEye":(153,0,153) , "rightEye":(102,0,153) , "leftEar": (153,0,50), "rightEar": (153,0,102),
                    "leftShoulder": (51,153,0), "rightShoulder": (153,102,0), "leftElbow": (0,153,0), "rightElbow": (153,153,0), "leftWrist": (0,153,51),
                    "rightWrist": (102,153,0), "leftHip":(0,51,153), "rightHip": (0,153,102), "leftKnee": (0,0,153), "rightKnee": (0,153,153),
                    "leftAnkle": (51,0,153), "rightAnkle": (0,102,153)}

    POSE_PAIRS = [['nose','leftEye'], ['nose','rightEye'], ['rightEye','rightEar'], ['leftEye','leftEar'],
                ['rightShoulder','rightElbow'], ['leftShoulder','leftElbow'], ['rightElbow','rightWrist'],
                ['leftElbow','leftWrist'], ['rightShoulder','leftShoulder'], ['rightShoulder','rightHip'],
                ['leftShoulder','leftHip'], ['rightHip','rightKnee'], ['leftHip','leftKnee'], ['rightKnee','rightAnkle'],
                ['leftKnee','leftAnkle'],['rightHip','leftHip']]

    interpreter = tf.lite.Interpreter('.\\posenet_mobilenet.tflite')

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    ############################################################ PERSON IMAGE ###################################################################

    frame = cv2.imread(ps)


    inHeight = input_details[0]['shape'][1]  #### AI model's Image height.
    inWidth = input_details[0]['shape'][2]   #### AI model's Image width.

    k = 0.57    ####### 
    kk = 0.3    #######
    frameHeight = frame.shape[0]  ##### the height of person's image
    frameWidth= frame.shape[1]    ##### the width of person's image
            
    inFrame = cv2.resize(frame, (inWidth, inHeight)) ##### Resize person's size into sample image's size.
    inFrame = cv2.cvtColor(inFrame, cv2.COLOR_BGR2RGB) ##### Change color system into sample image's type.

    inFrame = np.expand_dims(inFrame, axis=0) ##### expand dimension.
    inFrame = (np.float32(inFrame) - 127.5) / 127.5 ##### calculate every element's parameter.

    interpreter.set_tensor(input_details[0]['index'], inFrame)  ##### apply AI model.

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index']) ##### define output data of applied image
    offset_data = interpreter.get_tensor(output_details[1]['index']) ##### define output data of applied image

    heatmaps = np.squeeze(output_data) ##### remove dimension.
    offsets = np.squeeze(offset_data) ##### remove dimension.

    points = []
    joint_num = heatmaps.shape[-1]


    ####################################  MAKE PERSON'S POINTS ARRAY  ####################################
    for i in range(heatmaps.shape[-1]):   
        joint_heatmap = heatmaps[...,i]
        max_val_pos = np.squeeze(np.argwhere(joint_heatmap==np.max(joint_heatmap)))
        remap_pos = np.array(max_val_pos / 8 * 257,dtype=np.int32)
        point_y = int(remap_pos[0] + offsets[max_val_pos[0],max_val_pos[1],i])
        point_x = int(remap_pos[1] + offsets[max_val_pos[0],max_val_pos[1],i + joint_num])
        conf = np.max(joint_heatmap)

        x = (frameWidth * point_x) / inWidth
        y = (frameHeight * point_y) / inHeight

        points.append((int(x), int(y)) if conf > 0.3 else None)
        
    
    ############################################################ Detect area of the wearing part ################################################################### 


    roi2 = frame[int((points[0][1] + points[6][1]) / 2): int((points[8][1] + points[12][1]) * k), int((points[6][0] + points[8][0])/2):int((points[5][0] + points[7][0])/2)]
    sh, sw = roi2.shape[:2]

    
    cl_img = cv2.imread(cl) ##### Read Clothes image
    mh, mw = cl_img.shape[:2] ##### Calculate size of the clothes image.

 
    gray = cv2.cvtColor(cl_img, cv2.COLOR_BGR2GRAY) ##### change color system.
    if np.sum(cl_img[mh // 2][mw // 2]) > 600:  ##### If Clothes color is white.
        ret,th_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  ##### control threshold's value.
    else:
        ret,th_img = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) ##### control threshold's value.
       
        
    wnum = np.sum(th_img[int(sh / 2): int(sh / 2) + 1, 0: sw - 1] == 255) ##### calculate length of clothes's bottom line.
    rw = sw / wnum
    addw = int(sw * kk)

    #################################################### Match Clothes area and Person's wearing area ############################################
    cl_img = cv2.resize(cl_img, (sw + addw * 2, sh))
    th_img = cv2.resize(th_img, (sw + addw * 2, sh))


    th_img = cv2.cvtColor(th_img, cv2.COLOR_GRAY2BGR)
    can_img = cv2.Canny(cl_img,100,200)

    kernel = np.ones((9,9),np.uint8)
    can_img = cv2.morphologyEx(can_img, cv2.MORPH_CLOSE, kernel)
    
    ################################################### Process Clothes's neck ##################################################################
    contours, hierarchy = cv2.findContours(can_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if(area < sh * sw * 0.5):
            cv2.drawContours(th_img, [cnt], -1, (0, 0, 0), -1)
    
    ################################################### Overlay Clothe's part on Person's part ####################################################
    th_img = cv2.morphologyEx(th_img, cv2.MORPH_OPEN, kernel)
    frame[int((points[0][1] + points[6][1]) / 2): int((points[8][1] + points[12][1]) * k), int((points[6][0] + points[8][0])/2) - addw:int((points[5][0] + points[7][0])/2) + addw][th_img>0] = 0
    frame[int((points[0][1] + points[6][1]) / 2): int((points[8][1] + points[12][1]) * k), int((points[6][0] + points[8][0])/2) - addw:int((points[5][0] + points[7][0])/2) + addw]  += cl_img*(th_img>0)      
    
    ################################################### Save the result file into upload folder ########################################################
    res_path = settings.MEDIA_URL + "result.jpg"
    res = settings.MEDIA_ROOT_URL + res_path
    

    fs = FileSystemStorage()
    cv2.imwrite(res, frame)
    res_url = fs.url("result.jpg")
    ################################################## Get the Django's url of result file and return it ######################################################
    return res_url
         
