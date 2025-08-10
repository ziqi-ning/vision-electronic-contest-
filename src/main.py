#!/usr/bin/python3
# -*- coding: utf-8 -*-
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




vision_serialport=serial.Serial("/dev/ttyAMA4",115200,timeout=0.5)
vision_serialport.close()
vision_serialport.open()
if vision_serialport.isOpen():
    print("serialport init success")
    print(vision_serialport.port)
    print(vision_serialport.baudrate)
else:
    print("serialport init fail")

font=cv.FONT_HERSHEY_SIMPLEX
#cap = cv.VideoCapture(0)
cap = cv.VideoCapture("/dev/video0")
#cap.set(cv.CAP_PROP_EXPOSURE,4)
#cap.set(cv.CAP_PROP_AUTO_EXPOSURE,5000)
cap.set(cv.CAP_PROP_FPS,30)
cap.set(cv.CAP_PROP_FRAME_WIDTH,640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT,480)
ret,prev = cap.read()
img_flip=cv.flip(prev,0,None)
prev=img_flip.copy()
IMAGE_WIDTH=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
IMAGE_HEIGHT=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
class mode_ctrl(object):
    work_mode = 0x01 #工作模式.默认是点检测，可以通过串口设置成其他模式
    check_show = 1   #开显示，在线调试时可以打开，离线使用请关闭，可提高计算速度
ctr=mode_ctrl()



class target_check(object):
    x=0          #int16_t
    y=0          #int16_t
    pixel=0      #uint16_t
    flag=0       #uint8_t
    state=0      #uint8_t
    angle=0      #int16_t
    distance=0   #uint16_t
    apriltag_id=0#uint16_t
    img_width=0  #uint16_t
    img_height=0 #uint16_t
    reserved1=0  #uint8_t
    reserved2=0  #uint8_t
    reserved3=0  #uint8_t
    reserved4=0  #uint8_t
    fps=0        #uint8_t
    range_sensor1=0#uint8_t
    range_sensor2=0#uint8_t
    range_sensor3=0#uint8_t
    range_sensor4=0#uint8_t
    camera_id=0    #uint8_t
    reserved1_u32=0#uint32_t
    reserved2_u32=0#uint32_t
    reserved3_u32=0#uint32_t
    reserved4_u32=0#uint32_t
target=target_check()
target.camera_id=0x02
target.reserved1_u32=65536
target.reserved2_u32=105536
target.reserved3_u32=65537
target.reserved4_u32=105537

HEADER=[0xFF,0xFC]
MODE=[0xF1,0xF2,0xF3]
#__________________________________________________________________
def package_blobs_data(mode):
    #数据打包封装
    data=bytearray([HEADER[0],HEADER[1],0xA0+mode,0x00,
                   target.x>>8,target.x&0xff,        #将整形数据拆分成两个8位
                   target.y>>8,target.y&0xff,        #将整形数据拆分成两个8位
                   target.pixel>>8,target.pixel&0xff,#将整形数据拆分成两个8位
                   target.flag,                 #数据有效标志位
                   target.state,                #数据有效标志位
                   target.angle>>8,target.angle&0xff,#将整形数据拆分成两个8位
                   target.distance>>8,target.distance,#将整形数据拆分成两个8位
                   target.apriltag_id>>8,target.apriltag_id,#将整形数据拆分成两个8位
                   target.img_width>>8,target.img_width&0xff,    #将整形数据拆分成两个8位
                   target.img_height>>8,target.img_height&0xff,  #将整形数据拆分成两个8位
                   target.fps,      #数据有效标志位
                   target.reserved1,#数据有效标志位
                   target.reserved2,#数据有效标志位
                   target.reserved3,#数据有效标志位
                   target.reserved4,#数据有效标志位
                   target.range_sensor1>>8,target.range_sensor1&0xff,
                   target.range_sensor2>>8,target.range_sensor2&0xff,
                   target.range_sensor3>>8,target.range_sensor3&0xff,
                   target.range_sensor4>>8,target.range_sensor4&0xff,
                   target.camera_id,
                   target.reserved1_u32>>24&0xff,target.reserved1_u32>>16&0xff,
                   target.reserved1_u32>>8&0xff,target.reserved1_u32&0xff,
                   target.reserved2_u32>>24&0xff,target.reserved2_u32>>16&0xff,
                   target.reserved2_u32>>8&0xff,target.reserved2_u32&0xff,
                   target.reserved3_u32>>24&0xff,target.reserved3_u32>>16&0xff,
                   target.reserved3_u32>>8&0xff,target.reserved3_u32&0xff,
                   target.reserved4_u32>>24&0xff,target.reserved4_u32>>16&0xff,
                   target.reserved4_u32>>8&0xff,target.reserved4_u32&0xff,
                   0x00])
    #数据包的长度
    data_len=len(data)
    #print("serial data length is:",data_len)
    data[3]=data_len-5#有效数据的长度
    #和校验
    sum=0
    for i in range(0,data_len-1):
        sum=sum+data[i]
    data[data_len-1]=(sum&0xff)
    #返回打包好的数据
    return data
#__________________________________________________________________

#串口数据解析
def Receive_Anl(data_buf,num):
    #和校验
    sum = 0
    i = 0
    while i<(num-1):
        sum = sum + data_buf[i]
        i = i + 1
    sum = sum%256 #求余
    if sum != data_buf[num-1]:
        return
    #和校验通过
    if data_buf[2]==0xA0:
        #设置模块工作模式
        ctr.work_mode = data_buf[4]
        print("receive:{}".format(data_buf))
        print("Set work mode success!,work_mode is:",ctr.work_mode)

#__________________________________________________________________
class uart_buf_prase(object):
    uart_buf = []
    _data_len = 0
    _data_cnt = 0
    state = 0
R=uart_buf_prase()

def uart_data_prase(buf):
    if R.state==0 and buf==0xFF:#帧头1
        R.state=1
        R.uart_buf.append(buf)
    elif R.state==1 and buf==0xFE:#帧头2
        R.state=2
        R.uart_buf.append(buf)
    elif R.state==2 and buf<0xFF:#功能字
        R.state=3
        R.uart_buf.append(buf)
    elif R.state==3 and buf<50:#数据长度小于50
        R.state=4
        R._data_len=buf  #有效数据长度
        R._data_cnt=buf+5#总数据长度
        R.uart_buf.append(buf)
    elif R.state==4 and R._data_len>0:#存储对应长度数据
        R._data_len=R._data_len-1
        R.uart_buf.append(buf)
        if R._data_len==0:
            R.state=5
    elif R.state==5:
        R.uart_buf.append(buf)
        R.state=0
        Receive_Anl(R.uart_buf,R.uart_buf[3]+5)
        R.uart_buf=[]#清空缓冲区，准备下次接收数据
    else:
        R.state=0
        R.uart_buf=[]#清空缓冲区，准备下次接收数据

def uart_data_read():
    buf_len=vision_serialport.in_waiting
    if buf_len>0:
        buf=vision_serialport.read(buf_len)
        print("serial buf :",buf)
        for i in range(0,buf_len):
            uart_data_prase(buf[i])
#__________________________________________________________________



def get_cam_state():
    fps = cap.get(cv.CAP_PROP_FPS) 
    size=((int(cap.get(cv.CAP_PROP_FRAME_WIDTH))),(int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))
    exposure = cap.get(cv.CAP_PROP_EXPOSURE)
    gain = cap.get(cv.CAP_PROP_GAIN) 
    print(fps,size,exposure,gain,cv.getTickCount())


red_lower=np.array([156,43,43])
red_upper=np.array([180,255,255])
green_lower=np.array([35,43,43])
green_upper=np.array([77,255,255])
blue_lower=np.array([100,43,43])
blue_upper=np.array([124,255,255])
#lower_green = np.array([35, 50, 100])  # 设定绿色的阈值下限
#upper_green = np.array([77, 255, 255])  # 设定绿色的阈值上限
mybuffer = 10  #64
pts = deque(maxlen=mybuffer)

def opencv_find_color_blob(imgsrc):
    target.flag=0
    if (ctr.work_mode&0x05)!=0:
        hsv=cv.cvtColor(imgsrc,cv.COLOR_BGR2HSV)
        mask=cv.inRange(hsv,red_lower,red_upper)
        mask=cv.erode(mask,None,iterations=2)
        mask=cv.dilate(mask,None,iterations=2)
        _cnt,hierarchy=cv.findContours(mask.copy(),cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
        #cv.drawContours(imgsrc,_cnt,-1,(0,0,255),1)
        #cv.imshow("contour",imgsrc)
        center=None
        if(len(_cnt)>0):
            c=max(_cnt,key=cv.contourArea)
            _x,_y,_w,_h=cv.boundingRect(np.array(c))
            cv.rectangle(imgsrc,(_x,_y),(_x+_w,_y+_h),(0,0,255,10),2)
            pixels_max=cv.contourArea(c)
            ((x,y),radius)=cv.minEnclosingCircle(c)
            #print("x,y:",(x,y))
            M=cv.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
 
            if radius >= 3:  
                cv.circle(imgsrc, (int(x), int(y)), int(radius), (0, 255, 255), 2) 
                cv.circle(imgsrc, center, 5, (0, 0, 255), -1)
                pts.appendleft(center)
            for i in range(1, len(pts)):  
                if pts[i - 1] is None or pts[i] is None:  
                    continue
                thickness = int(np.sqrt(mybuffer / float(i + 1)) * 1.0)
                cv.line(imgsrc, pts[i - 1], pts[i], (0, 0, 255), thickness)
            x=int(x)
            y=int(y)
            text='('+str_format(x)+","+str_format(y)+')'
            cv.putText(imgsrc,text,(x,y),font,0.5,(0, 0, 255),1)

            target.img_width=IMAGE_WIDTH
            target.img_height=IMAGE_HEIGHT
            target.x=x
            target.y=y
            target.pixel=int(pixels_max/100)
            target.flag=1
        # cv.imshow("result", imgsrc)


class camera_params(object):
    fx=0
    fy=0
    cx=0
    cy=0
    tag_size_m=0
cam_info=camera_params()
cam_info.tag_size_m=0.15    #15cm-0.15
cam_info.fx=(3.6/3.6736)*IMAGE_WIDTH#4.2
cam_info.fy=(3.6/2.7384)*IMAGE_HEIGHT
cam_info.cx=IMAGE_WIDTH/2
cam_info.cy=IMAGE_HEIGHT/2
info_t=[cam_info.fx,cam_info.fy,cam_info.cx,cam_info.cy]
print(info_t)
apriltag_detetor=apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))

def opencv_find_april_tag(imgsrc):
    target.flag=0
    if (ctr.work_mode&0x04)!=0:
        gray=cv.cvtColor(imgsrc,cv.COLOR_BGR2GRAY)
        tags,img_=apriltag_detetor.detect(gray,return_image=True)
        # cv.imshow("apriltag", img_)
        for tag in tags:
            (ptA,ptB,ptC,ptD)=tag.corners
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

            # cv.circle(imgsrc,ptA,3,(255,0,0),1)#left-top
            # cv.circle(imgsrc,ptB,3,(0,0,255),1)#right-top
            # cv.circle(imgsrc,ptC,3,(0,0,255),1)#right-buttom
            # cv.circle(imgsrc,ptD,3,(0,0,255),1)#left-bottom
            # cv.line(imgsrc,ptA,ptB,(255,0,0),1,cv.LINE_AA)#AB
            # cv.line(imgsrc,ptB,ptC,(0,255,0),1,cv.LINE_AA)#BC
            # cv.line(imgsrc,ptC,ptD,(0,0,255),1,cv.LINE_AA)#CD
            # cv.line(imgsrc,ptD,ptA,(255,0,255),1,cv.LINE_AA)#DA
            # cv.putText(imgsrc,'A'+str_format(ptA)+"   AB:"+str_format(len_ab),ptA,font,0.5,(0, 0, 255),1)
            # cv.putText(imgsrc,'B'+str_format(ptB)+"   BC:"+str_format(len_bc),ptB,font,0.5,(0, 0, 255),1)
            # cv.putText(imgsrc,'C'+str_format(ptC)+"   CD:"+str_format(len_cd),ptC,font,0.5,(0, 0, 255),1)
            # cv.putText(imgsrc,'D'+str_format(ptD)+"   AD:"+str_format(len_da),ptD,font,0.5,(0, 0, 255),1)
            (x,y)=tuple(tag.center.astype(int))
            # cv.putText(imgsrc,'O'+str_format((x,y))+"  ID:"+str_format(tag.tag_id),(x,y),font,0.5,(0, 0, 255),1)

            #print("side len is:",side_len,pixels_max)
            target.img_width=IMAGE_WIDTH
            target.img_height=IMAGE_HEIGHT
            target.x=x
            target.y=y
            target.apriltag_id=tag.tag_id
            target.pixel=side_len
            target.flag=1 
            target.reserved1_u32=int(P_Mat[0]*1000)#单位为mm
            target.reserved2_u32=int(P_Mat[1]*1000)#单位为mm
            target.reserved3_u32=int(P_Mat[2]*1000)#单位为mm
            
    # cv.imshow("result", imgsrc)



def decodeDisplay(_img):
    target.flag=0
    image=cv.cvtColor(_img,cv.COLOR_BGR2GRAY)
    IMAGE_WIDTH=int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    IMAGE_HEIGHT=int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    target.img_width=IMAGE_WIDTH
    target.img_height=IMAGE_HEIGHT
    barcodes = pyzbar.decode(image)
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
        target.x=int((2*x+w)/2)
        target.y=int((2*y+h)/2)
        target.apriltag_id=int(barcodeData)
        target.pixel=0
        target.flag=1
        #print(barcode.rect,target.x,target.y,target.img_width,target.img_height)
    return _img



exec_dt_ms=0
img_update_flag=0
pre_tick=0
cur_tick=0
def fps_capture():
    global curr,ret,prev,exec_dt_ms
    global pre_tick,cur_tick,img_update_flag
    pre_tick=cur_tick
    cur_tick=cv.getTickCount()
    exec_dt_ms=1000*(cur_tick-pre_tick)/cv.getTickFrequency()
    target.fps=int(1000/exec_dt_ms)


def auto_capture():
    t=Timer(0.1,auto_capture)
    t.start()
     #一帧一帧捕捉
    #ret,curr = cap.read()
    #img_flip=cv.flip(curr,-1,None)#1:hor 0:vert -1:hor+vert
    #curr=img_flip.copy()
    #prev=curr
    #img_update_flag=1
    #fps_capture()
    #print("demo")



get_cam_state()
auto_capture()

ctr.work_mode=0x00

while(True):
    ret,curr = cap.read()
    if ret==True:
        img_flip=cv.flip(curr,-1,None)#1:hor 0:vert -1:hor+vert
        curr=img_flip.copy()
        img_update_flag=1
        # 显示返回的每帧
        if img_update_flag==1:
            img_update_flag=0
            if ctr.work_mode==0x00:#空闲模式
                print(target.fps,"fps")
            elif ctr.work_mode==0x01:#空闲模式
                print(target.fps,"fps")
            elif ctr.work_mode==0x02:#空闲模式
                print(target.fps,"fps")
            elif ctr.work_mode==0x03:#空闲模式
                print(target.fps,"fps")
            elif ctr.work_mode==0x04:#AprilTag模式
                opencv_find_april_tag(curr)
                print(target.fps,"fps")
            elif ctr.work_mode==0x05:#色块模式
                opencv_find_color_blob(curr)
                print(target.fps,"fps")
            elif ctr.work_mode==0x06:#条形码
                result=decodeDisplay(curr)
                # 显示图片
                # cv.imshow("result", result)
                #获取系统时间并格式化输出显示

                print(target.fps,"fps")
            elif ctr.work_mode==0x07:#预留模式3
                print(target.fps,"fps")
            else:
                print(target.fps,"fps")
            print("current work mode is:",ctr.work_mode)
            # print("flag:{}".format(target.flag))
            vision_serialport.write(package_blobs_data(ctr.work_mode))
            uart_data_read()
            fps_capture()
#______________________________________________________________
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
# 当所有事完成，释放 VideoCapture 对象
cap.release()
cv.destroyAllWindows()

