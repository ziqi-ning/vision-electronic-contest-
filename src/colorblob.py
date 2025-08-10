import cv2 as cv
import numpy as np
from collections import deque
from numpy.core.arrayprint import str_format

font=cv.FONT_HERSHEY_SIMPLEX

red_lower = np.array([156, 43, 43])
red_upper = np.array([180, 255, 255])
green_lower = np.array([35, 43, 43])
green_upper = np.array([77, 255, 255])
blue_lower = np.array([100, 43, 43])
blue_upper = np.array([124, 255, 255])
black_lower = np.array([42, 0, 0])
black_upper = np.array([127, 255, 94])
red_laser_lower = np.array([156, 43, 43])
red_laser_upper = np.array([180, 255, 255])
all_lower = np.array([0, 57, 0])
all_upper = np.array([180, 255, 255])
tiger_lower = np.array([0, 71, 0])
tiger_upper = np.array([34, 255, 255])
wolf_lower = np.array([0, 49, 0])
wolf_upper = np.array([129, 255, 108])
elephant_lower = np.array([0, 49, 0])
elephant_upper = np.array([180, 255, 173])
monkey_lower = np.array([0, 61, 0])
monkey_upper = np.array([32, 210, 172])
peacock_lower = np.array([0, 45, 0])
peacock_upper = np.array([164, 255, 255])



color_threshold = {"red": (red_lower, red_upper), 
                    "green": (green_lower, green_upper), 
                    "blue": (blue_lower, blue_upper),
                    "black": (black_lower, black_upper),
                    "red_laser": (red_laser_lower, red_laser_upper),
                    "all": (all_lower, all_upper),
                    "tiger": (tiger_lower, tiger_upper),
                    "wolf": (wolf_lower, wolf_upper),
                    "elephant": (elephant_lower, elephant_upper),
                    "monkey": (monkey_lower, monkey_upper),
                    "peacock": (peacock_lower, peacock_upper)
                    }


mybuffer = 10  #64
pts = deque(maxlen=mybuffer) #坐标中心队列，可能用于预测


def detect_color(imgsrc, str_color, bais=20, min_area=1200):
    '''
    输入参数：
    imgsrc：输入图像
    str_color：颜色，"red"、"green"、"blue"
    bais：范围偏移量，指定要检测颜色区域的增加范围，默认30，识别直线需缩小
    返回值为字典列表：
    result：输出图像，包含检测到的颜色区域，其余为黑色
    x：中心点x坐标
    y：中心点y坐标
    pixels_max：面积
    注意：列表为空需在函数外手动添加黑色掩模
    '''

    # 要返回的先声明好
    x = 0
    y = 0
    pixels_max = 0
    result = imgsrc
    center=None

    # 提取对应颜色的阈值
    lower, upper = color_threshold[str_color]

    # 转为HSV
    hsv=cv.cvtColor(imgsrc,cv.COLOR_BGR2HSV)

    # 生成掩模
    mask=cv.inRange(hsv,lower,upper)
    # cv.imshow("src_mask",mask)

    # 闭运算
    mask=cv.erode(mask,None,iterations=2)
    mask=cv.dilate(mask,None,iterations=2)
    # cv.imshow("mask_deal",mask)

    # 寻找区域轮廓
    _cnt,hierarchy=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(imgsrc,_cnt,-1,(0,0,255),1)
    #cv.imshow("contour",imgsrc)

    _cnt_fliter = [cnt for cnt in _cnt if cv.contourArea(cnt) > min_area]

    cnt_efc = []

    # 创建满足区域颜色阈值的三通道版本掩码
    mask2_3channel = cv.merge([mask, mask, mask])

    result = cv.bitwise_and(imgsrc, mask2_3channel)

    for c in _cnt_fliter:

        # 提取最小包围圆参数
        ((x,y),radius)=cv.minEnclosingCircle(c)
        radius = int(radius)
        #print("x,y:",(x,y))

        # 得到矩
        M=cv.moments(c)

        # 通过矩得到质心
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        if radius >= 3:  
            # 创建全黑掩码（单通道）
            mask = np.zeros(imgsrc.shape[:2], dtype=np.uint8)
            
            # 在掩码上绘制圆形边框
            cv.circle(mask, center, radius+bais, 255, -1)

            # 创建三通道版本掩码
            mask_3channel = cv.merge([mask, mask, mask])
            
            # 按位与操作保留圆形区域
            result = cv.bitwise_and(imgsrc, mask_3channel)
            # cv.imshow("roi_mask",result)
            pixels_max = cv.countNonZero(mask)
            cnt_efc.append({"result": result, "center": center, "pixels_max": pixels_max})

    # 按面积整理列表
    cnt_efc = sorted(cnt_efc, key=lambda x: x["pixels_max"], reverse=True)
        # 画中心点测试    
        # for i in range(1, len(pts)):  
        #     if pts[i - 1] is None or pts[i] is None:  
        #         continue
        #     thickness = int(np.sqrt(mybuffer / float(i + 1)) * 1.0)
        #     cv.line(imgsrc, pts[i - 1], pts[i], (0, 0, 255), thickness)
        # x=int(x)
        # y=int(y)
        # text='('+str_format(x)+","+str_format(y)+')'
        # cv.putText(imgsrc,text,(x,y),font,0.5,(0, 0, 255),1)
    
    # print("bais:{}".format(bais))
    return cnt_efc
    

def detect_color_to_rect(imgsrc, str_color, bais=20, min_area=1200):
    '''
    输入参数：
    imgsrc：输入图像
    str_color：颜色，"red"、"green"、"blue"
    bais：范围偏移量，指定要检测颜色区域的增加范围，默认30，识别直线需缩小
    返回值为字典列表：
    result：输出图像，包含检测到的颜色区域，其余为黑色
    x：中心点x坐标
    y：中心点y坐标
    pixels_max：面积
    注意：列表为空需在函数外手动添加黑色掩模
    '''

    # 要返回的先声明好
    x = 0
    y = 0
    pixels_max = 0
    result = imgsrc
    center=None

    # 提取对应颜色的阈值
    lower, upper = color_threshold[str_color]

    # 转为HSV
    hsv=cv.cvtColor(imgsrc,cv.COLOR_BGR2HSV)

    # 生成掩模
    mask=cv.inRange(hsv,lower,upper)
    # cv.imshow("src_mask",mask)

    # 闭运算
    mask=cv.erode(mask,None,iterations=2)
    mask=cv.dilate(mask,None,iterations=2)
    # cv.imshow("mask_deal",mask)

    # 寻找区域轮廓
    _cnt,hierarchy=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(imgsrc,_cnt,-1,(0,0,255),1)
    #cv.imshow("contour",imgsrc)

    _cnt_fliter = [cnt for cnt in _cnt if cv.contourArea(cnt) > min_area]

    cnt_efc = []

    # 创建满足区域颜色阈值的三通道版本掩码
    mask2_3channel = cv.merge([mask, mask, mask])

    result = cv.bitwise_and(imgsrc, mask2_3channel)

    for c in _cnt_fliter:

        if len(c) > 4: 
            # 提取最小包围矩形参数
            rect = cv.minAreaRect(c)
            box = cv.boxPoints(rect)
            box = np.array(box, dtype="int")
            box = expand_box(box, 1.2)
            box = range_site(box)
            #print("box:",box)

            # 得到矩
            M=cv.moments(c)

            # 通过矩得到质心
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

            # 创建全黑掩码（单通道）
            mask = np.zeros(imgsrc.shape[:2], dtype=np.uint8)
            
            # 在掩码上绘制矩形边框
            cv.drawContours(mask, [box], 0, 255, -1)

            # 创建三通道版本掩码
            mask_3channel = cv.merge([mask, mask, mask])
            
            # 按位与操作保留矩形区域
            result = cv.bitwise_and(imgsrc, mask_3channel)
            # cv.imshow("roi_mask",result)
            pixels_max = cv.countNonZero(mask)
            cnt_efc.append({"result": result, "center": center, "pixels_max": pixels_max})

    # 按面积整理列表
    cnt_efc = sorted(cnt_efc, key=lambda x: x["pixels_max"], reverse=True)
        # 画中心点测试    
        # for i in range(1, len(pts)):  
        #     if pts[i - 1] is None or pts[i] is None:  
        #         continue
        #     thickness = int(np.sqrt(mybuffer / float(i + 1)) * 1.0)
        #     cv.line(imgsrc, pts[i - 1], pts[i], (0, 0, 255), thickness)
        # x=int(x)
        # y=int(y)
        # text='('+str_format(x)+","+str_format(y)+')'
        # cv.putText(imgsrc,text,(x,y),font,0.5,(0, 0, 255),1)
    
    # print("bais:{}".format(bais))
    return cnt_efc


def detect_multi_color(imgsrc, str_color1, str_color2, bais=20, min_area=1200):
    '''
    输入参数：
    imgsrc：输入图像
    str_color：颜色，"red"、"green"、"blue"
    bais：范围偏移量，指定要检测颜色区域的增加范围，默认30，识别直线需缩小
    返回值为字典列表：
    result：输出图像，包含检测到的颜色区域，其余为黑色
    x：中心点x坐标
    y：中心点y坐标
    pixels_max：面积
    注意：列表为空需在函数外手动添加黑色掩模
    '''

    # 要返回的先声明好
    x = 0
    y = 0
    pixels_max = 0
    result = imgsrc
    center=None

    # 提取对应颜色的阈值
    lower1, upper1 = color_threshold[str_color1]
    lower2, upper2 = color_threshold[str_color2]

    # 转为HSV
    hsv=cv.cvtColor(imgsrc,cv.COLOR_BGR2HSV)

    # 生成掩模
    mask1=cv.inRange(hsv,lower1,upper1)
    mask2=cv.inRange(hsv,lower2,upper2)
    # cv.imshow("src_mask",mask)

    mask=cv.bitwise_or(mask1,mask2)

    # 闭运算
    mask=cv.erode(mask,None,iterations=2)
    mask=cv.dilate(mask,None,iterations=2)
    # cv.imshow("mask_deal",mask)

    # 寻找区域轮廓
    _cnt,hierarchy=cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(imgsrc,_cnt,-1,(0,0,255),1)
    #cv.imshow("contour",imgsrc)

    _cnt_fliter = [cnt for cnt in _cnt if cv.contourArea(cnt) > min_area]

    cnt_efc = []

    # 创建满足区域颜色阈值的三通道版本掩码
    mask2_3channel = cv.merge([mask, mask, mask])

    result = cv.bitwise_and(imgsrc, mask2_3channel)

    for c in _cnt_fliter:

        # 提取最小包围圆参数
        ((x,y),radius)=cv.minEnclosingCircle(c)
        radius = int(radius)
        #print("x,y:",(x,y))

        # 得到矩
        M=cv.moments(c)

        # 通过矩得到质心
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

        if radius >= 3:  
            # 创建全黑掩码（单通道）
            mask = np.zeros(imgsrc.shape[:2], dtype=np.uint8)
            
            # 在掩码上绘制圆形边框
            cv.circle(mask, center, radius+bais, 255, -1)

            # 创建三通道版本掩码
            mask_3channel = cv.merge([mask, mask, mask])
            
            # 按位与操作保留圆形区域
            result = cv.bitwise_and(imgsrc, mask_3channel)
            # cv.imshow("roi_mask",result)
            pixels_max = cv.countNonZero(mask)
            cnt_efc.append({"result": result, "center": center, "pixels_max": pixels_max})

    # 按面积整理列表
    cnt_efc = sorted(cnt_efc, key=lambda x: x["pixels_max"], reverse=True)
        # 画中心点测试    
        # for i in range(1, len(pts)):  
        #     if pts[i - 1] is None or pts[i] is None:  
        #         continue
        #     thickness = int(np.sqrt(mybuffer / float(i + 1)) * 1.0)
        #     cv.line(imgsrc, pts[i - 1], pts[i], (0, 0, 255), thickness)
        # x=int(x)
        # y=int(y)
        # text='('+str_format(x)+","+str_format(y)+')'
        # cv.putText(imgsrc,text,(x,y),font,0.5,(0, 0, 255),1)
    
    # print("bais:{}".format(bais))
    return cnt_efc

def range_site(myPoints):
    """
    用于整顿矩形ROI的角点顺序,让它合法(来自22届的祖传函数，非常好用)
    """
    #print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)] # 左上角和最大
    myPointsNew[2] = myPoints[np.argmax(add)] # 右下角和最小
    diff = np.diff(myPoints,axis=1)#会自动大减小及不可能出现负值所以行差值最小的会是第二个点这个可以重数学角度去理解
    myPointsNew[3] = myPoints[np.argmin(diff)] # 左下角差最小
    myPointsNew[1] = myPoints[np.argmax(diff)] # 右上角差最大
    return myPointsNew

def expand_box(box, scale=1.2):
    """按比例扩大矩形ROI框：scale>1时向外扩展，<1时向内收缩"""
    center = np.mean(box, axis=0)  # 计算中心点
    expanded_box = center + (box - center) * scale  # 所有点沿中心外推
    return expanded_box.astype(int)

def detect_laser(imgsrc, light_bais=20, min_area=0, max_area=500):
    '''
    输入：
    imgsrc：输入图像
    bais：范围偏移量，指定要检测激光区域的浮动范围，默认20
    min_area：最小面积，默认800
    返回：
    flag：是否检测到激光，1为检测到，0为未检测到
    result_img：输出图像
    center：中心点坐标
    '''
    frame = imgsrc.copy()
    flag = 0
    center = (0, 0)
    radius = 0
    if imgsrc is None:
        return flag, center, radius

    # 转为HSV
    img_hsv = cv.cvtColor(imgsrc, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(img_hsv)

    max_val = np.max(v)
    # max_val = 255
    # 使用阈值将高亮度区域二值化（保留最大亮度附近±bais的范围）
    _, thresh = cv.threshold(v, max_val - light_bais, 255, cv.THRESH_BINARY)

    # 形态学操作（消除噪点）
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
    opened = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)

    # cv.imshow("thresh", opened)

    # 寻找轮廓
    contours, _ = cv.findContours(opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cv.drawContours(frame, contours, -1, (0, 255, 0), 2)
    cv.imshow("contours", frame)

    max_contour = None
    midle_area = 0

    # 筛选符合面积要求的轮廓
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > min_area and area < max_area and area > midle_area:
            midle_area = area
            max_contour = cnt
            
    img_test = cv.drawContours(imgsrc, max_contour, -1, (0, 255, 0), 2)
    cv.imshow("max_contour", img_test)
    print("midle_area:", midle_area)

    if max_contour is not None:
        # 用最小外接圆替代矩形包围盒
        flag = 1
        (circle_x, circle_y), radius = cv.minEnclosingCircle(max_contour)
        center = (int(circle_x), int(circle_y))
        radius = int(radius)
        result_img = cv.circle(imgsrc, center, radius, (0, 255, 0), 2)

        return flag, result_img, center, radius
    
    return flag, imgsrc, center, radius