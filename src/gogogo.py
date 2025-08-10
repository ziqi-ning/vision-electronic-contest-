#!/usr/bin/env python3
import cv2
import numpy as np  
import serial
import asyncio
import time
import outsite
import facility2
import uartuse
import colorblob
import other
import apriltag
import radar5
import allin
from collections import deque
from numpy.core.arrayprint import str_format


colors = ["empty", "white", "red", "yellow", "green", "indigo", "blue", "purple"]

distance = 3000

target = uartuse.TargetCheck()
ctr = uartuse.ModeCtrl()
R = uartuse.UartBufParse()


red_lower = np.array([156, 43, 43])
red_upper = np.array([180, 255, 255])
green_lower = np.array([35, 43, 43])
green_upper = np.array([77, 255, 255])
blue_lower = np.array([100, 43, 43])
blue_upper = np.array([124, 255, 255])

pts = deque(maxlen=10)


queue_pictures = asyncio.Queue(maxsize=2)
queue_serial_info = asyncio.Queue(maxsize=5) # 存放接受到的控制信息对象
queue_radar_test = asyncio.Queue(maxsize=1) # 存放雷达测定图片
queue_radar_draw = asyncio.Queue(maxsize=1)

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

fx=(3.6/3.6736)*IMAGE_WIDTH  #4.2 水平焦距
fy=(3.6/2.7384)*IMAGE_HEIGHT # 垂直焦距

cx=IMAGE_WIDTH/2 # 光心x
cy=IMAGE_HEIGHT/2 # 光心y

delta_x = 0       # X方向偏移（右侧为正）
delta_y = 0.09   # Y方向偏移（前方为负）雷达在相机后方0.09米
delta_z = -0.12   # Z方向偏移（上方为负）雷达在相机上方0.12米，实际飞机是-0.18，这里便于测试
camera_pitch_deg = 0  # 相机俯仰角度（度）

camera_height = 0.03  # 相机高度（米）

angle_tolerance_rad = 0.5  # ±2.86°容差（约3°）


camera_params = (
    fx, fy, cx, cy, 
    delta_x, delta_y, delta_z,
    camera_pitch_deg, angle_tolerance_rad,
    camera_height
)


class static_camera_params(object):
    fx=0
    fy=0
    cx=0
    cy=0
    tag_size_m=0

cam_info=static_camera_params()
cam_info.tag_size_m=0.15    #15cm-0.15
cam_info.fx=(3.6/3.6736)*IMAGE_WIDTH#4.2
cam_info.fy=(3.6/2.7384)*IMAGE_HEIGHT
cam_info.cx=IMAGE_WIDTH/2
cam_info.cy=IMAGE_HEIGHT/2

apriltag_detetor=apriltag.Detector(apriltag.DetectorOptions(families='tag36h11'))
QR_detector = cv2.QRCodeDetector()


exec_dt_ms=0
img_update_flag=0
pre_tick=0
cur_tick=0


async def measure(radar_manager, camera_params, goal, angle_tolerance_rad=10*100) :
    x = goal[0]
    y = goal[1]
    center = (int(x), int(y))
    center2 = (int(x), int(y)+20)

    try:
        frame = queue_radar_test.get_nowait()
    except asyncio.QueueEmpty:
        await asyncio.sleep(0.01)

    cv2.circle(frame, center, 2, (0, 0, 255), thickness=1, lineType=8)

    # 坐标转距离
    distance, angle = await radar_manager.site_to_distance(x, y, camera_params)
    cv2.putText(frame, "distance: {:.2f}cm, angle={:.3f}".format(distance,angle), center, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # 得到障碍物列表
    after_deal = await radar_manager.get_obstacle()
    print("angel_measure: {},{}".format(distance, angle))
    
    # 得到角度差列表,angles和after_deal索引一一一对应
    angles = [item[1] for item in after_deal]

    # 候选点列表,格式(距离, 角度)
    candidate = []

    left_diffs = []
    right_diffs = []
    closest_index = -1

    # 添加自己
    candidate.append((distance, angle))

    # 此处的i是after_deal的索引,angels也能用
    for i, item in enumerate(angles):
        angle_diff = abs(angle - item)
        if angle_diff <= 18000:  # 180度内
            left_diffs.append((angle_diff, i))
        else:  # 跨越360度边界的情况
            right_diffs.append((36000 - angle_diff, i))

    # 整理左右角度差列表，从小到大整理
    left_diffs = sorted(left_diffs, key=lambda x: x[0])
    right_diffs = sorted(right_diffs, key=lambda x: x[0])

    for left in left_diffs[:2]:
        if left[0] < angle_tolerance_rad:
            candidate.append(after_deal[left[1]])
            print("left")

    for right in right_diffs[:2]:
        if right[0] < angle_tolerance_rad:
            candidate.append(after_deal[right[1]])
            print("right")


    # # 根据角度差判断是否添加进候选点集
    # if min_left_diff[0] < angle_tolerance_rad:
    #     candidate.append(after_deal[min_left_diff[1]])
    #     print("left")

    # if min_right_diff[0] < angle_tolerance_rad:
    #     candidate.append(after_deal[min_right_diff[1]])
    #     print("right")



    closest_obstacle = min(candidate, key=lambda x: x[0])
    closest_angle = closest_obstacle[1]
    closest_range = closest_obstacle[0]
    
    print("----------------------------------------------------")
    print(candidate)
    print("obstacle_measure: {},{}".format(closest_range, closest_angle))
    cv2.putText(frame, "near: {:.2f}cm, angle:{}".format(closest_range, closest_angle), center2, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if queue_radar_draw.full():
        queue_radar_draw.get_nowait()
    await queue_radar_draw.put(frame)
    await asyncio.sleep(0.1)



async def fps_capture():
    global curr,ret,prev,exec_dt_ms
    global pre_tick,cur_tick,img_update_flag
    pre_tick=cur_tick
    cur_tick=cv2.getTickCount()
    exec_dt_ms=1000*(cur_tick-pre_tick)/cv2.getTickFrequency()
    if int(1000/exec_dt_ms) > 255:
        target.fps = 255
    else:
        target.fps=int(1000/exec_dt_ms)




async def serial_get(reader):
    print("serial has been ready")
    while True:
        line = await reader.read(1)
        if line:
            # print("receive:{}".format(line))
            uartuse.uart_data_prase(R, line[0], ctr)
        await asyncio.sleep(0)  # 立即让出控制权
    


async def video_get(cam):
    while True:
        ret,frame = cam.read()
        frame = cv2.flip(frame, 0, None)
        if not ret:
            print("ret wrong")                               
        if queue_pictures.full():                          # 队列满了删除最早的帧
            queue_pictures.get_nowait()
        await queue_pictures.put(frame) 
        #print("放入成功一个")                                     
        #print("Waiting for video data...")
        await asyncio.sleep(0)


#______________________________________________________________
async def deal_data(ser, radar_manager):
    
    ctr.work_mode=0x00 # 初始空闲模式
    count = 0

    fps_avg = 0
    fps_last = 0
    fps_pub = [0]

    while(True):
        try:
            frame = queue_pictures.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.01)
            continue

        img_update_flag=1  # 图像更新标志
        target.flag = 0
        # 显示返回的每帧

        target.img_width=IMAGE_WIDTH
        target.img_height=IMAGE_HEIGHT

        if img_update_flag==1:
            img_update_flag=0
            if ctr.work_mode==0x00:#空闲模式
                    # 获取当前事件循环
                cnt_efc = colorblob.detect_color_to_rect(frame, "red")
                if cnt_efc:
                    result_img = cnt_efc[0]["result"]
                    center = cnt_efc[0]["center"]
                    if queue_radar_test.full():
                        queue_radar_test.get_nowait()
                    await queue_radar_test.put(result_img)
                    loop = asyncio.get_running_loop()
                    # 创建任务但不阻塞
                    loop.create_task(measure(radar_manager, camera_params, center))
                    try:
                        radar_draw = queue_radar_draw.get_nowait()
                    except asyncio.QueueEmpty:
                        await asyncio.sleep(0.01)
                        continue
                    if radar_draw is not None:
                        cv2.imshow("radar_draw", radar_draw)
            elif ctr.work_mode==0x01:# 识别圆的模式
                print("work_mode_one")
            elif ctr.work_mode==0x02:#听声辩位雷达模式
                pass
            elif ctr.work_mode==0x03:#空闲模式
                pass
            elif ctr.work_mode==0x04:#AprilTag模式
                pass
            elif ctr.work_mode==0x05:#色块模式
                pass
            elif ctr.work_mode==0x06:#条形码
                pass
            elif ctr.work_mode==0x07:#24年真题
                img, flag, data, center, pixel = other.QR_detect(QR_detector, frame)
                target.flag=flag
                target.apriltag_id = data
                target.x = center[0]
                target.y = center[1]
                target.pixel = pixel
            else:
                pass

            # 标志位指示灯
            if target.flag==1:
                count+=1
                if count>10:
                    count=0
                    await light.setColor("indigo")
            else:
                color = colors[ctr.work_mode]
                await light.setColor(color)

            # 调试串口
            # print("serial has sent:\nd:{}\nh:{}".format(list(package), package.hex()))
            
            # 平均帧率
            fps_last = fps_pub.pop()
            fps_avg = int((target.fps+fps_last)/2)
            fps_pub.append(fps_avg)

            # 调试信息
            # print("fps_avg:{}".format(fps_avg))
            # print("flag:{}".format(target.flag))
            # print("work_mode:{}".format(ctr.work_mode))


            # cv2.imshow("frame", img)

            # 捕捉帧率去了
            await fps_capture()

            # 读模式码
            uartuse.uart_data_read(R, ser, ctr)

            # 串口包装发送
            package = uartuse.package_blobs_data(ctr.work_mode, target)
            ser.write(package)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
#______________________________________________________________



async def main():

    try:
        ser=serial.Serial("/dev/ttyAMA4",256000,timeout=0.5)
        ser.close()
        ser.open()
        if ser.isOpen():
            print("serialport init success")
            print(ser.port)
            print(ser.baudrate)
        else:
            print("serialport init fail")
            exit()
    except Exception as e:
        print("serialport error")
        print(e)
        exit()

    try:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        cam.set(cv2.CAP_PROP_FPS, 50)
        if not cam.isOpened():
            print("Cannot open camera")
            exit()
    except Exception as e:
        print("camera error")
        print(e)
        exit()

    try:
        radar_manager = radar5.RadarManager()
    except:
        print("radar error")
        exit()

    tasks = [video_get(cam), deal_data(ser, radar_manager)]
    await asyncio.gather(*tasks)

if __name__ == '__main__':
    
    try:
        light = facility2.light()
    except Exception as e:
        print("GPIO error")
        print(e)
        exit()
    asyncio.run(main())


    