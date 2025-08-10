import pyzbar.pyzbar as pyzbar
from numpy.core.arrayprint import str_format
from os import truncate
from threading import Timer
from collections import deque
from numpy.core.arrayprint import str_format
from numpy.core.defchararray import center, lower
from numpy.lib.utils import info
import numpy as np
import math as mh
import cv2 as cv
import time
import serial
import apriltag
import pyzbar.pyzbar as pyzbar


    

def decodeDisplay(_img):
    flag = 0
    image=cv.cvtColor(_img,cv.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(image)
    x = 0
    y = 0
    apriltag_id = 0
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        (x, y, w, h) = barcode.rect
        cv.rectangle(_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # 绘出图像上条形码的数据和条形码类型
        text = "{} ({})".format(barcodeData, barcodeType)+str_format(barcode.rect)
        cv.putText(_img, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX,
                    .5, (0, 0, 255), 1)
        # 向终端打印条形码数据和条形码类型
        print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
        x=int((2*x+w)/2)
        y=int((2*y+h)/2)
        apriltag_id=int(barcodeData)
        flag = 1
        #print(barcode.rect,target.x,target.y,target.img_width,target.img_height)
    return _img, x, y, apriltag_id, flag


def opencv_find_april_tag(imgsrc, cam_info, apriltag_detetor):
    gray = cv.cvtColor(imgsrc,cv.COLOR_BGR2GRAY)
    tags = apriltag_detetor.detect(gray,return_image=False)
    for tag in tags:
        (ptA,ptB,ptC,ptD)=tag.corners
        info_t=[cam_info.fx,cam_info.fy,cam_info.cx,cam_info.cy]
        #print(tag.__str__)
        #print("tag id:",tag.tag_id)
        ptA=(int(ptA[0]),int(ptA[1]))
        ptB=(int(ptB[0]),int(ptB[1]))
        ptC=(int(ptC[0]),int(ptC[1]))
        ptD=(int(ptD[0]),int(ptD[1]))
        M,e1,e2=apriltag_detetor.detection_pose(tag,info_t,cam_info.tag_size_m)
        R_Mat=np.array(M[0:3,0:3])
        P_Mat=np.array(M[0:3,3])
        yaw=180*mh.atan2(R_Mat[1][0],R_Mat[0][0])/mh.pi
        pitch=180*mh.asin(R_Mat[2][0])/mh.pi
        roll=180*mh.atan2(R_Mat[2][1],R_Mat[2][2])/mh.pi
        #print(yaw,pitch,roll)
        #print("Rotate data is:",R_Mat)
        #print("Pose data is:",P_Mat)
        px=format(P_Mat[0],'.4f')
        py=format(P_Mat[1],'.4f')
        pz=format(P_Mat[2],'.4f')
        # print("position x y z is:",px,py,pz)
        len_ab=int(mh.sqrt(np.square(ptB[0]-ptA[0])+np.square(ptB[1]-ptA[1])))
        len_bc=int(mh.sqrt(np.square(ptB[0]-ptC[0])+np.square(ptB[1]-ptC[1])))
        len_cd=int(mh.sqrt(np.square(ptC[0]-ptD[0])+np.square(ptC[1]-ptD[1])))
        len_da=int(mh.sqrt(np.square(ptD[0]-ptA[0])+np.square(ptD[1]-ptA[1])))
        max_len=max(len_ab,len_bc)
        max_len=max(max_len,len_cd)
        max_len=max(max_len,len_da)
        side_len=max_len
        flag = 1

        (x,y)=tuple(tag.center.astype(int))

        return x, y, tag.tag_id, side_len, flag, int(P_Mat[0]*1000), int(P_Mat[1]*1000), int(P_Mat[2]*1000)
    return 0, 0, 0, 0, 0, 0, 0, 0


def QR_detect(detector, img):

    flag = 0
    data = 0

    # 检测并解码单个二维码
    data, points, _ = detector.detectAndDecode(img)

    # 如果检测成功
    if points is not None:
        # 绘制二维码边界框
        points = points[0].astype(int)
        
        # 显示解码结果
        if data:
            data = int(data)
                # 计算中心点
            center = tuple(np.mean(points, axis=0).astype(int))
            if center[0]<0 or center[1]<0:
                return img, flag, 0, (0,0), 0
            # 计算左侧边长

            flag = 1
            left_length = int(np.linalg.norm(points[0] - points[3]))
            x = int(center[0])
            y = int(center[1])
            pixel = int(left_length*left_length)
            
            # 在图像上标注中心和边长
            cv.circle(img, center, 5, (0, 255, 0), -1)  # 绘制中心点
            cv.putText(img, f"Center: {center}", (10, 30), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv.putText(img, f"Side Length: {left_length}px", (10, 70), 
                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 绘制二维码边框
            cv.polylines(img, [points], True, (0, 0, 255), 2)
            # 显示结果
            print("Decoded Data:", data)
            return img, flag, data, x, y, min(60000,pixel)
        
        return img, flag, 0, (0,0), 0
    else:
        # print("No QR code detected")
        return img, flag, 0, (0,0), 0
        


