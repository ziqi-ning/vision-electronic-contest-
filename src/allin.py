import cv2
import outsite
import colorblob
import numpy as np
from collections import Counter

def color_bonous_usual(frame, color):
    """
    此颜色ROI合成函数是通用模板,想识别具体的形状复制其进行修改
    输入：
    frame：待处理的图像
    color：目标颜色(str类型)

    输出：
    composite_img：直接合成图像
    """
    cnt_efc = colorblob.detect_color_to_rect(frame, color)

    composite_img = np.zeros(frame.shape, dtype=np.uint8)

    # 创建记录已填充区域的掩码（单通道）
    filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for item in cnt_efc:

        # 提取图像
        result_img = item['result']

        # 以此为模板添加你的识别处理，将会合成你的所有的信息

        # 创建当前图像的掩码（非黑色区域）
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 只处理未填充的区域
        fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))
        
        # 高效更新合成图像
        composite_img[fill_mask == 255] = result_img[fill_mask == 255]
        
        # 更新已填充区域掩码
        filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

    return composite_img

def find_type(imgsrc):
    """
    输入：三通道图像(一般只有一个ROI,因为这里只能单图像识别)
    输出:
    img：处理后的图像
    type_info：当前字典列表,存储当前识别形状
    """
    mode = 0
    type_info = []
    blur_contour = imgsrc.copy()
    if mode == 0:
        flag, img, trapezoid_info, center_max, width_max = outsite.detect_trapezoids(imgsrc)
        # print("flag:", flag)
        # print("center:", center_max)
        # print("width:", width_max)
        if flag == 1:
            # print("梯形")
            type_info.append({"type": 1, "center": center_max, "lengh": width_max})
        else:
            mode = 1
    if mode == 1:
        flag, img, triangles_info, center_max, radius_max = outsite.detect_triangle(imgsrc)
        # print("flag:", flag)
        # print("center:", center_max)
        # print("radius:", radius_max)
        if flag == 1:
            # print("三角形")
            type_info.append({"type": 2, "center": center_max, "lengh": radius_max})
        else:
            mode = 2
    if mode == 2:
        flag, img, pole_groups, center = outsite.find_longest_straight_line(imgsrc)
        # print("flag:", flag)
        # print("center:", center)
        if flag == 1:
            # print("杆子")
            type_info.append({"type": 3, "center": center, "lengh": 0})
        else:
            mode = 3
    if mode == 3:

        flag, img, ellipse_info, center, r_max = outsite.detect_ellipses(imgsrc)
        # print("flag:", flag)
        # print("center:", center)
        # print("r:", r_max)
        if flag == 1:
            # print("圆")
            type_info.append({"type": 0, "center": center, "lengh": r_max})
            
    return img, type_info

def give_me_a_color_and_i_will_give_you_a_shape(frame, color, bais):
    """
    输入：
    frame：待处理的图像
    color：目标颜色(str类型)

    输出：
    composite_img：查找各区域形状完毕，并合成的图像
    type_list：各区域形状信息
    """

    cnt_efc = colorblob.detect_color(frame, color, bais)

    composite_img = np.zeros(frame.shape, dtype=np.uint8)

    # 创建记录已填充区域的掩码（单通道）
    filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    type_list = []
    
    for item in cnt_efc:

        # 提取图像
        result_img = item['result']

        # 检测形状
        result_img, type_info = find_type(result_img)

        # 创建当前图像的掩码（非黑色区域）
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 只处理未填充的区域
        fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))
        
        # 高效更新合成图像
        composite_img[fill_mask == 255] = result_img[fill_mask == 255]
        
        # 更新已填充区域掩码
        filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

        type_list = type_list + type_info 

    shape_counts = Counter(item["type"] for item in type_list)

    # print(shape_counts)

    return composite_img, type_list

def blur_contour_only(src_img, contour, dilate_radius=5, blur_kernel=(25,25)):
    """仅在轮廓线周围生成带状模糊区域
    
    Args:
        src_img: 原图（三通道BGR）
        contour: 单个轮廓（np.array格式）
        dilate_radius: 边缘扩展半径(像素)，控制模糊带宽度
        blur_kernel: 高斯模糊核，必须为奇数
    """
    # 创建线宽1像素的轮廓掩模
    mask = np.zeros_like(src_img[:,:,0])
    cv2.drawContours(mask, [contour], -1, 255, thickness=1)  # 仅绘制1像素宽的线
    
    # 扩展轮廓线区域（关键步骤）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dilate_radius+1,)*2)
    expanded_mask = cv2.dilate(mask, kernel)
    
    # 整体模糊后按条件替换
    blurred = cv2.GaussianBlur(src_img, blur_kernel, 0)
    
    # 三维蒙版处理（重要！避免通道不匹配）
    condition = expanded_mask[:,:,None].astype(bool)  # 升维至[h,w,1]
    result = np.where(condition, blurred, src_img)
    
    return result.astype(np.uint8)



def color_bonous_ORB(frame, color):
    """
    ORB专用色块合成器
    输入：
    frame：待处理的图像
    color：目标颜色(str类型)

    输出：
    composite_img：直接合成图像
    """
    # 寻找色块列表
    cnt_efc = colorblob.detect_color(frame, color, bais = 0)

    # 创建黑色幕布
    composite_img = np.zeros(frame.shape, dtype=np.uint8)

    # 创建记录已填充区域的掩码（单通道）
    filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for item in cnt_efc:

        # 提取图像
        result_img = item['result']
        
        # 检测圆形RO带来的边框,以便模糊方式角点误检测
        test_img, contour = outsite.detect_ellipse_max_one(result_img)

        # cv2.imshow("test_img", test_img)

        # 模糊处理,防止角点误检测
        if contour is not None:
            result_img = blur_contour_only(result_img, contour, dilate_radius=5, blur_kernel=(25,25))

        # 创建当前图像的掩码（非黑色区域）
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 只处理未填充的区域
        fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))
        
        # 高效更新合成图像
        composite_img[fill_mask == 255] = result_img[fill_mask == 255]
        
        # 更新已填充区域掩码
        filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

    return composite_img


def color_bonous_for_line(frame, color):
    """
    寻找直线专用色块合成器
    输入：
    frame：待处理的图像
    color：目标颜色(str类型)

    输出：
    composite_img：直接合成图像
    """
    # 老三样,色块列表
    cnt_efc = colorblob.detect_color_to_rect(frame, color)

    # 创建黑色幕布
    composite_img = np.zeros(frame.shape, dtype=np.uint8)

    # 创建记录已填充区域的掩码（单通道）
    filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for item in cnt_efc:

        # 提取图像
        result_img = item['result']

        # 查找直线并绘图
        flag, result_img, pole_groups, center = outsite.find_longest_straight_line(result_img)


        # 创建当前图像的掩码（非黑色区域）
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 只处理未填充的区域
        fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))
        
        # 高效更新合成图像
        composite_img[fill_mask == 255] = result_img[fill_mask == 255]
        
        # 更新已填充区域掩码
        filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

    return composite_img


def match_indicater(good_match, type, kp0, kp1):
    """
    此匹配方式供单模板使用
    输入参数：
    good_mathch：匹配结果对
    type：分为0和1两种模式，0模式为单应性矩阵验证，1模式为平均匹配距离验证
    kp0：模板图像关键点
    kp1：待匹配图像关键点

    输出参数：
    indicater：是否匹配成功
    """
    if len(good_match) < 10 :
        return False
    else:
        if type == 0:
            if len(good_match) > 4:

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
                dst_pts = np.float32([kp0[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = mask.ravel().tolist().count(1)
                
                if inliers/len(good_match) > 0.7:
                    # print(f"单应性矩阵验证成功，内点数量：{inliers}/{len(good_match)}")
                    return True
                else:
                    return False
                
            return  False
        
        elif type == 1:
            avg_distance = sum([m.distance for m in good_match]) / len(good_match)
            if avg_distance < 30:
                #　print(f"平均匹配距离：{avg_distance:.2f}")
                return True
            else:
                return False

        else:
            print("模式错误")
            return False
        


def singel_match(good_match, type, kp0, kp1):
    """
    此匹配方式供多模板，寻找最相近使用
    输入参数：
    good_mathch：匹配结果对
    type：分为0和1两种模式，0模式为单应性矩阵验证，1模式为平均匹配距离验证
    kp0：模板图像关键点
    kp1：待匹配图像关键点

    输出参数：
    indicater：是否匹配成功，此处一般不用
    value：匹配结果参数。1模式越大越准，0模式越小越准
    """

    if len(good_match) < 10 :
        return False, 0
    else:
        if type == 0:
            if len(good_match) > 4:

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good_match]).reshape(-1,1,2)
                dst_pts = np.float32([kp0[m.trainIdx].pt for m in good_match]).reshape(-1,1,2)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                inliers = mask.ravel().tolist().count(1)
                
                if inliers/len(good_match) > 0.6:
                    # print(f"单应性矩阵验证成功，内点数量：{inliers}/{len(good_match)}")
                    return True, inliers/len(good_match)
                else:
                    return False, inliers/len(good_match)
                
            return  False, 0
        
        elif type == 1:
            avg_distance = sum([m.distance for m in good_match]) / len(good_match)
            if avg_distance < 50:
                # print(f"平均匹配距离：{avg_distance:.2f}")
                return True, avg_distance
            else:
                return False, avg_distance

        else:
            print("模式错误")
            return False, 0

def template_matching(kk, frame, color):
    """
    模板匹配器(遍历多模板专用)
    输入参数：
    kk：模板图像
    frame：待匹配图像
    color：目标颜色(str类型)

    输出参数：
    indicater：是否匹配成功
    outimage：匹配结果图像
    score：匹配结果参数。1模式越大越准，0模式越小越准
    """

    # 归一化尺寸
    kk = cv2.resize(kk, (640, 480))

    # 创建ORB对象和匹配器
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    #　提取模板图像关键点
    kp0 = orb.detect(kk, None)

    # 计算模板图像描述子
    kp0, des0 = orb.compute(kk, kp0)

    # print(kp0)

    # 容器,防assigh错误
    candidate = None
    good_match = []

    # composite_img, type_list = allin.give_me_a_color_and_i_will_give_you_a_shape(frame)

    # 寻找色块,合成后的图像
    candidate = color_bonous_ORB(frame, color)

    # 描述符
    kp1, des1 = orb.detectAndCompute(candidate, None)

    if des1 is not None and des0 is not None:

        # 开始匹配啦
        matches = bf.match(des1, des0)

        # 按质量排序
        good_match = sorted(matches, key=lambda x:x.distance)[:45]

    # 绘制匹配结果
    outimage = cv2.drawMatches(candidate, kp1, kk, kp0, good_match, outImg=None)

    # print("num of good match: ", len(good_match))
    # 调用我们的大函数
    indicater, score = singel_match(good_match, 1, kp0, kp1)

    if indicater is True:
        # print("yes")
        return indicater, outimage, score
    else:
        # print("no")
        return indicater, outimage, score
    

def color_bonous_multi_color(frame, color1, color2):
    """
    此颜色ROI合成函数是通用多颜色模板,想识别具体的形状复制其进行修改
    输入：
    frame：待处理的图像
    color：目标颜色(str类型)

    输出：
    composite_img：直接合成图像
    """
    cnt_efc = colorblob.detect_multi_color(frame, color1, color2)

    composite_img = np.zeros(frame.shape, dtype=np.uint8)

    # 创建记录已填充区域的掩码（单通道）
    filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for item in cnt_efc:

        # 提取图像
        result_img = item['result']

        # 以此为模板添加你的识别处理，将会合成你的所有的信息

        # 创建当前图像的掩码（非黑色区域）
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 只处理未填充的区域
        fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))
        
        # 高效更新合成图像
        composite_img[fill_mask == 255] = result_img[fill_mask == 255]
        
        # 更新已填充区域掩码
        filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

    return composite_img



def color_bonous_laser_small_area(frame, color, bais=5, min_area=500):
    """
    此颜色ROI合成函数是识别激光专用合成图像函数,想识别具体的形状复制其进行修改
    输入：
    frame：待处理的图像
    color：目标颜色(str类型)

    输出：
    composite_img：直接合成图像
    """
    cnt_efc = colorblob.detect_color(frame, color, bais=bais, min_area=min_area)

    composite_img = np.zeros(frame.shape, dtype=np.uint8)

    # 创建记录已填充区域的掩码（单通道）
    filled_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    for item in cnt_efc:

        # 提取图像
        result_img = item['result']

        # 以此为模板添加你的识别处理，将会合成你的所有的信息

        # 创建当前图像的掩码（非黑色区域）
        gray = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
        _, current_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 只处理未填充的区域
        fill_mask = cv2.bitwise_and(current_mask, cv2.bitwise_not(filled_mask))
        
        # 高效更新合成图像
        composite_img[fill_mask == 255] = result_img[fill_mask == 255]
        
        # 更新已填充区域掩码
        filled_mask = cv2.bitwise_or(filled_mask, fill_mask)

    return composite_img


def template_matching_animals(kk, candidate):
    """
    模板匹配器(遍历多模板专用)
    输入参数：
    kk：模板图像
    frame：待匹配图像
    color：目标颜色(str类型)

    输出参数：
    indicater：是否匹配成功
    outimage：匹配结果图像
    score：匹配结果参数。1模式越大越准，0模式越小越准
    """
    if candidate is None:
        return False, None, 0

    # 归一化尺寸
    kk = cv2.resize(kk, (640, 480))
    candidate = cv2.resize(candidate, (640, 480))

    # 创建ORB对象和匹配器
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    #　提取模板图像关键点
    kp0 = orb.detect(kk, None)

    # 计算模板图像描述子
    kp0, des0 = orb.compute(kk, kp0)

    # print(kp0)

    good_match = []

    # composite_img, type_list = allin.give_me_a_color_and_i_will_give_you_a_shape(frame)

    # 描述符
    kp1, des1 = orb.detectAndCompute(candidate, None)

    if des1 is not None and des0 is not None:

        # 开始匹配啦
        matches = bf.match(des1, des0)

        # 按质量排序
        good_match = sorted(matches, key=lambda x:x.distance)[:45]

    # 绘制匹配结果
    outimage = cv2.drawMatches(candidate, kp1, kk, kp0, good_match, outImg=None)

    # print("num of good match: ", len(good_match))
    # 调用我们的大函数
    indicater, score = singel_match(good_match, 1, kp0, kp1)

    if indicater is True:
        # print("yes")
        return indicater, outimage, score
    else:
        # print("no")
        return indicater, outimage, score
