import cv2
import numpy as np
import math
import allin


def detect_ellipse_max_one(image, canny_threshold=20, min_contour_area=250, aspect_ratio_tol=0.2, start_area=250):
    """
    检测图像中的最大椭圆。（经过圆形ROI掩模之后的镜头）主要用途是排除ROI带来的额外角点影响匹配
    
    参数：
    image : 输入图像（BGR格式）
    canny_threshold : Canny边缘检测阈值（默认20）
    min_contour_area : 最小轮廓面积阈值（默认100）
    aspect_ratio_tol : 椭圆纵横比容差（0-1，默认0.2）
    start_area : 敲门砖椭圆面积阈值（默认250）
    
    返回：
    result_img,：带标记的原图，作测试用
    contour_max：轮廓，用于allin的blur_contour_only模糊边缘
    此函数用作识别ROI的圆框，不作真正的识别圆用。
    使用后搭配模糊边缘。
    """
    if image is None:
        return image, None

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯滤波降噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    
    # Canny边缘检测
    edges = cv2.Canny(blurred, canny_threshold//2, canny_threshold)    
    # cv2.imshow("edges", edges)


    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = image.copy()
    max_ellipse = None
    max_area = start_area
    r = 0

    contour_max = None

    for contour in contours:
        # 过滤小面积轮廓
        if cv2.contourArea(contour) < min_contour_area:
            continue
            
        # 至少需要5个点来拟合椭圆
        if contour.shape[0] < 5:
            continue
            
        try:
            ellipse = cv2.fitEllipse(contour)
            
            # 计算椭圆参数
            (x, y), (a, b), angle = ellipse
            current_area = np.pi * a * b
            r = math.sqrt(a * b)
            
            # 过滤不合理的椭圆（纵横比容差）
            if abs(a - b)/max(a, b) > aspect_ratio_tol:
                continue
                
            # 更新最大椭圆
            if current_area > max_area:
                max_area = current_area
                max_ellipse = ellipse
                contour_max = contour
                
            # 绘制所有有效椭圆（绿色）
            cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)
            
        except:
            continue

    # 绘制最大椭圆（红色）
    if max_ellipse is not None:
        # 绘制椭圆
        cv2.ellipse(result_img, max_ellipse, (0, 0, 255), 4)
        # 绘制中心点
        center = tuple(map(int, max_ellipse[0]))
        cv2.circle(result_img, center, 5, (255, 255, 0), 8)
        # 添加文字标注
        cv2.putText(result_img, "Max Ellipse", (center[0]-50, center[1]-15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    else:
        center = (0, 0) 


    return result_img, contour_max


def detect_ellipses(image, canny_threshold=20, min_contour_area=250, bais_area= 0.2, start_area=250, ratio_limit = 0.5):
    """
    检测图像中的椭圆，绘制轮廓和椭圆，并返回最大椭圆的参数
    
    参数：
    image : 输入图像（BGR格式）
    canny_threshold : Canny边缘检测阈值（默认20）
    min_contour_area : 最小轮廓面积阈值（默认100）
    bais_area : 轮廓与拟合椭圆的面积偏差百分比
    start_area : 敲门砖椭圆面积阈值（默认250）
    ratio_limit : a与b的比例偏差，在0~1之间。用于过滤不合理的椭圆（默认0.5）
    返回：
    result_img : 绘制了椭圆的结果图像
    max_ellipse : 最大椭圆的参数 ((中心坐标), (轴长), 旋转角度)
    center : 最大椭圆的中心坐标
    r : 最大椭圆的半径

    警告：未检测到时返回 (0, 0)，调用千万注意
    这里排除了小轮廓，需要就调阈值
    """

    flag = 0
    result_img = image
    ellipse_info = []
    center = (0, 0)
    r_max = 0

    if image is None:
        return flag, result_img, ellipse_info, center, r_max

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 高斯滤波降噪
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    
    # Canny边缘检测
    edges = cv2.Canny(blurred, canny_threshold//2, canny_threshold)    
    # cv2.imshow("edges", edges)


    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        # 过滤小面积轮廓
        if cv2.contourArea(contour) < min_contour_area:
            continue
            
        # 至少需要5个点来拟合椭圆
        if contour.shape[0] < 5:
            continue
            
        try:
            ellipse = cv2.fitEllipse(contour)

            contour_area = cv2.contourArea(contour)
            
            # 计算椭圆参数
            (x, y), (a, b), angle = ellipse
            current_area = np.pi * a * b
            r = math.sqrt(a * b)
            
            # area_ratio = abs(contour_area - current_area) / contour_area

            # # 面积偏差太多不视作椭圆
            # if area_ratio > bais_area:
            #     continue
            if a == 0 or b == 0:
                continue

            if min(a/b, b/a) < ratio_limit:
                continue

            ellipse_info.append({
                "center": (x, y),
                "axes": (a, b),
                "angle": angle,
                "r": r,
                "area": current_area,
                "ellipse": ellipse
            })
                
            # if current_area > max_area:
            #     max_area = current_area
            #     max_ellipse = ellipse
                
            # 绘制所有有效椭圆（绿色）
            cv2.ellipse(result_img, ellipse, (0, 255, 0), 2)
            
        
        except:
            continue

    sorted_ellipse_info = sorted(ellipse_info, key=lambda info: info["area"], reverse=True)

    # 最终判断

    # 定义阈值：用于判断中心距离和半径差异的阈值
    radius_diff_threshold = 0.1    # 半径差异阈值（百分比），可根据需求调整


    if len(sorted_ellipse_info) == 0:
        return flag, result_img, ellipse_info, center, r_max
    # 初始化变量
    largest_ellipse = sorted_ellipse_info[0]  # 最大椭圆
    second_largest_ellipse = None  # 初始化第二大椭圆为 None

    # 判断第二大椭圆是否与最大椭圆接近
    for idx in range(1, len(sorted_ellipse_info)):
        current_ellipse = sorted_ellipse_info[idx]  # 当前椭圆

        # 计算半径的差异（可以使用面积的平方根作为近似的半径）
        max_radius = np.sqrt(largest_ellipse["area"] / np.pi)
        current_radius = np.sqrt(current_ellipse["area"] / np.pi)
        radius_difference = abs(max_radius - current_radius) / max_radius

        # 判断中心距离和半径差异是否接近
        if radius_difference <= radius_diff_threshold:
            # 如果接近，跳过当前椭圆，继续判断下一个椭圆
            continue
        else:
            # 如果不接近，当前椭圆是真正意义上的第二大椭圆
            second_largest_ellipse = current_ellipse
            break

    # 如果找到了真正的第二大椭圆，绘制相关信息
    if second_largest_ellipse is not None:
        flag = 1
        # 绘制椭圆
        cv2.ellipse(result_img, second_largest_ellipse["ellipse"], (0, 0, 255), 4)
        # 绘制中心点
        center = tuple(map(int, second_largest_ellipse["center"]))
        r_max = int(second_largest_ellipse["r"])
        cv2.circle(result_img, center, 5, (255, 255, 0), 8)
        # 添加文字标注
        cv2.putText(result_img, "True Second Largest Ellipse", (center[0] - 50, center[1] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    else:
        r_max = 0
        center = (0, 0)

    return flag, result_img, ellipse_info, center, r_max


def angle_cos(p1, p2, p0):
    """计算三个点之间的角度余弦值"""
    d1, d2 = (p1 - p0).astype(np.float32), (p2 - p0).astype(np.float32)
    return np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2))


def detect_trapezoids(img, max_area = 10, cos_max = 0.85, min_contour_area = 100):
    # 读取图像
    '''
    检测图像中的梯形，绘制轮廓，并返回最大梯形的参数
    参数：
    image : 输入图像（BGR格式）
    min_contour_area: 最小轮廓面积阈值（默认100）
    max_area:最大梯形面积阈值
    cos_max:最大余弦值阈值，超过则视作直线
    返回：
    img:勾勒后图像
    trapezoid_info:所有梯形的字典
    center_max:最大梯形中心坐标
    width_max:最大梯形的宽
    这里没有排除小轮廓
    '''
    flag = 0
    trapezoid_info = []
    center = (0, 0)
    center_max = (0, 0)
    width_max = 0

    if img is None:
        print("Error: Image not loaded")
        return flag, img, trapezoid_info, center_max, width_max
    
    # 1. 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 中值滤波 (5x5核)
    blurred = cv2.medianBlur(gray, 5)
    
    # 3. Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. 膨胀操作 (使用3x3核)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # cv2.imshow("Dilated", dilated)
    
    # 5. 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    trapezoids = []
    # 面积太小的进行过滤

    for cnt in contours:

        if cv2.contourArea(cnt) < min_contour_area:
            continue


        # 6. 多边形逼近 (精度=轮廓周长3%)
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 7. 检测4边形且为凸多边形
        if len(approx) == 4 and cv2.isContourConvex(approx):
            # 计算最大内角余弦值（排除接近直线的退化情况）
            cosines = []
            for i in range(4):
                p0, p1, p2 = approx[i][0], approx[(i+1)%4][0], approx[(i+2)%4][0]
                cos = angle_cos(p1, p2, p0)
                cosines.append(cos)
            
            # 排除内角过小的退化四边形（余弦值接近±1）
            max_cos = max(np.abs(cosines))
            if max_cos < cos_max:  # 阈值可调整
                approx = approx.reshape(-1, 2)
                trapezoids.append(approx)
                area = cv2.contourArea(approx)
                if area > max_area: # 更新最大梯形
                    flag = 1
                    max_area = area
                    # 去除冗余维度后直接操作
                    max_trap = cnt.reshape(-1, 2)
                    center_max = (int(np.mean(max_trap[:, 0])), int(np.mean(max_trap[:, 1])))
                    rect_max = cv2.minAreaRect(max_trap)
                    (w_max, h_max) = rect_max[1]
                    width_max = int(min(w_max, h_max))
                    

    for approx in trapezoids:
        # 顶点坐标（假设是4x2的numpy数组）
        approx = approx.reshape(-1, 2)
        
        # 计算中心
        center = (int(np.mean(approx[:, 0])), int(np.mean(approx[:, 1])))
        
        # 计算宽度（以最小包围矩形为例）
        rect = cv2.minAreaRect(approx)
        (w, h) = rect[1]
        
        # 合并结果
        trapezoid_info.append({
            "contour": approx, 
            "center": center,
            "width": int(min(w, h))  # 取较小者为宽度
        })
    for trap in trapezoids:
            cv2.polylines(img, [trap.reshape(-1, 1, 2).astype(int)], True, (0, 255, 0), 3)
            center_x = int(np.mean(trap[:, 0]))
            center_y = int(np.mean(trap[:, 1]))
            cv2.putText(img, "Trapezoid", (center_x - 50, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return flag, img, trapezoid_info, center_max, width_max


def detect_triangle(img, max_area = 10, cos_max = 0.85, min_contour_area = 100):
    '''
    检测图像中的三角形，绘制轮廓，并返回最大三角形的参数
    参数：
    image : 输入图像（BGR格式）
    min_contour_area: 最小轮廓面积阈值（默认100）
    max_area:最大三角形初始面积阈值
    cos_max:最大余弦值阈值，超过则视作直线
    返回：
    img:勾勒后图像
    triangles_info:所有三角形的字典
    center_max:最大三角形中心坐标
    radius_max:最大三角形的宽
    这里没有排除小轮廓
    '''
    # 读取图像
    flag = 0
    triangles_info = []
    center = (0, 0)
    center_max = (0, 0)
    radius = 0
    radius_max = 0

    if img is None:
        print("Error: Image not loaded")
        return flag, img, triangles_info, center_max, radius_max
    
    # 1. 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 中值滤波 (5x5核)
    blurred = cv2.medianBlur(gray, 9)
    
    # 3. Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    
    # 4. 膨胀操作 (使用3x3核)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # cv2.imshow("Dilated", dilated)
    
    # 5. 寻找轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    


    triangles = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_contour_area:
            continue
        # 6. 多边形逼近 (精度=轮廓周长3%)
        perimeter = cv2.arcLength(cnt, True)
        epsilon = 0.03 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # 7. 检测4边形且为凸多边形
        if len(approx) == 3 and cv2.isContourConvex(approx):
            # 计算最大内角余弦值（排除接近直线的退化情况）
            cosines = []
            for i in range(3):
                p0, p1, p2 = approx[i][0], approx[(i+1)%3][0], approx[(i+2)%3][0]
                cos = angle_cos(p1, p2, p0)
                cosines.append(cos)
            
            # 排除内角过小的退化情形（余弦值接近±1）
            max_cos = max(np.abs(cosines))
            if max_cos < cos_max:  # 阈值可调整
                approx = approx.reshape(-1, 2)
                triangles.append(approx)
                area = cv2.contourArea(approx)
                if area > max_area:# 更新最大三角形
                    flag = 1
                    max_area = area
                    # 去除冗余维度后直接操作
                    max_trap = cnt.reshape(-1, 2)
                    (center_max, radius) = cv2.minEnclosingCircle(max_trap)
                    radius_max = int(radius)
                    center_max = (int(center_max[0]), int(center_max[1]))


    for approx in triangles:
        # 顶点坐标（假设是4x2的numpy数组）
        approx = approx.reshape(-1, 2)
        
        # 计算中心
        (center, radius) = cv2.minEnclosingCircle(approx)
        radius = int(radius)
        
        # 合并结果
        triangles_info.append({
            "contour": approx, 
            "center": center, 
            "radius": radius  # 取较小者为宽度
        })

    for trap in triangles:
            cv2.polylines(img, [trap.reshape(-1, 1, 2).astype(int)], True, (0, 255, 0), 3)
            center_x = int(np.mean(trap[:, 0]))
            center_y = int(np.mean(trap[:, 1]))
            cv2.putText(img, "Trapezoid", (center_x - 50, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return flag, img, triangles_info, center_max, radius_max


def calculate_line_distance(line1, line2):
    """计算两条平行直线之间的平均距离"""
    # 转换为直线方程：Ax + By + C = 0
    x1, y1, x2, y2 = line1
    A1 = y2 - y1
    B1 = x1 - x2
    C1 = x2*y1 - x1*y2
    
    # 取四个端点计算平均距离
    points = [line2[:2], line2[2:]]
    distances = []
    for x, y in points:
        numerator = abs(A1*x + B1*y + C1)
        denominator = np.sqrt(A1**2 + B1**2)
        distances.append(numerator / denominator)
    
    return np.mean(distances)


def find_longest_straight_line(image, vertical_angle_threshold = 10, min_line_length=100, max_gap=20):
    """
    在图像中找出最长且最直的线条
    
    参数:
    image: 输入图像
    min_line_length: 线段最小长度阈值
    max_gap: 线段最大间隙
    vertical_angle_threshold: 与垂直方向的角度阈值，太水平的不要
    
    返回:
    lines: 检测到的线段端点坐标列表
    result_image: 绘制线条后的结果图像
    """

    pole_groups = []
    flag = 0


    if image is None:
        print("Error: Image not loaded")
        return flag, image, pole_groups, 0

    # 1. 图像预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray,5)
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # 2. 使用霍夫变换检测线条
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, 
                            minLineLength=min_line_length, 
                            maxLineGap=max_gap)
    
    # 3. 如果没有检测到线条，返回空结果
    if lines is None:
        # print("没有检测到符合条件的线条")
        return flag, image, pole_groups, 0
    
    # 修改后的候选线段处理逻辑（在原代码位置修改）
    # 候选列表改为字典结构，按角度分组（5度为一个区间）
    angle_groups = {}  # {角度区间: {'lines': 线段列表, 'total_quality': 总质量}}

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1


        cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)

        theta = np.arctan2(dy, dx) * 180 / np.pi
        if theta < 0:
            theta += 180  # 转换到 0° 到 180° 范围
        if theta > 180:
            theta -= 180  # 转换到 0° 到 180° 范围
        
        # 计算线段与垂直方向的夹角
        vertical_angle = np.abs(np.abs(theta) - 90)

        if vertical_angle > vertical_angle_threshold:
            continue


        length = np.sqrt(dx**2 + dy**2)
        
        # 计算线段角度（0-180度范围）
        angle = np.arctan2(dy, dx) * 180 / np.pi
        if angle < 0:
            angle += 180
        
        # 角度分组（5度为一个区间）
        angle_key = round(angle / 3) * 3
        
        points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)

        # 质量计算（长度 + 直线度）
        quality = length * 0.8 + cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)[-1] * 0.2
        
        # 维护角度分组
        if angle_key not in angle_groups:
            angle_groups[angle_key] = {'lines': [], 'total_quality': 0}
        
        group = angle_groups[angle_key]
        if len(group['lines']) < 2:  # 每组最多保留2条最佳线段
            group['lines'].append((quality, line[0], angle))
            group['total_quality'] += quality
            # 组内按质量排序
            group['lines'].sort(reverse=True, key=lambda x: x[0])
        else:
            if quality > group['lines'][-1][0]:
                group['total_quality'] -= group['lines'][-1][0]
                group['lines'].pop()
                group['lines'].append((quality, line[0], angle))
                group['total_quality'] += quality
                group['lines'].sort(reverse=True, key=lambda x: x[0])

    # print("groups_num:{}".format(len(angle_groups)))
    # 按分组总质量排序，保留前3组最佳平行线组
    best_groups = sorted(angle_groups.values(), key=lambda x: -x['total_quality'])[:20]


    for group in best_groups:
        if len(group['lines']) >= 2:
            # 计算组内线段平均间距
            line1 = group['lines'][0][1]
            line2 = group['lines'][1][1]
            avg_distance = calculate_line_distance(line1, line2)
            
            # 假设杆子间距在5-20像素之间
            if 5 < avg_distance < 70:
                flag = 1
                    # 计算杆子组内所有线段的中点的平均 x 坐标
                x_coords = [line[1][0] for line in group['lines']]  # 提取所有线段的 x1 和 x2
                x2_coords = [line[1][2] for line in group['lines']]  # 提取所有线段的 x2
                
                # 计算所有线段中点的 x 坐标
                avg_x = np.mean([(x1 + x2) / 2 for x1, x2 in zip(x_coords, x2_coords)])
                
                pole_groups.append({
                    'angle': group['lines'][0][2],
                    'lines': [line[1] for line in group['lines']],
                    'avg_distance': avg_distance,
                    'center': [int(avg_x)]
                })

    # 作图
    color_anlysis = (0, 0, 255)

    for pole in pole_groups:
        count = 0
        for line in pole['lines']:
            x1, y1, x2, y2 = line
            # 同一组平行线段内红蓝交替作图
            if count%2 == 0:
                color_anlysis = (0, 0, 255)
            else:
                color_anlysis = (255, 0, 0)
            count += 1
            # print("count:{}".format(count))
            cv2.line(image, (x1,y1), (x2,y2), color_anlysis, 2)
        for center in pole['center']:
            cv2.line(image, (center, 0), (center, 470), (0, 255, 0), 2)
    # print("pole_groups:{}".format(len(pole_groups)))

    if pole_groups:
        return flag, image, pole_groups, pole_groups[0]["center"]
    else:
        return flag, image, pole_groups, 0



