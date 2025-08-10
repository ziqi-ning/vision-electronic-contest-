"""此文件仅作测试：测试阈值对Canny轮廓检测效果的影响"""
import cv2
import numpy as np

"""定义卷积核"""
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))   # 椭圆核：用于形态学处理

"""辅助函数"""
def nothing(x):
    pass

"""原始图像 - 灰度图像 - 图像滤波(双边/高斯) - 二值化(边缘检测/阈值化) - 形态学处理"""
if __name__ == '__main__':
    """初始化"""
    # 【1】摄像头初始化
    cam_width = 640
    cam_height = 480
    cap = cv2.VideoCapture(0)   # 获取摄像头的指针(传入摄像头的序号)
    # 自动获取帧的宽度和高度
    cam_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # 手动设置摄像头分辨率及帧率
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)
    # cap.set(cv2.CAP_PROP_FPS, 60)
    # 【2】Canny边缘检测
    low = 100     # Canny低阈值：threshold1
    high = 200    # Canny高阈值：threshold2
    # 【3】创建公共窗口以及滑动条(调节Canny阈值)
    cv2.namedWindow('Canny Edge Demo')
    # 创建滑动条
    cv2.createTrackbar('low', 'Canny Edge Demo', 100, 255, nothing)
    cv2.createTrackbar('high', 'Canny Edge Demo', 200, 255, nothing)



    """识别循环"""
    while True:
        '''获取图像'''
        ret, img = cap.read()  # 每次循环都通过capture指针的read函数来实时获取摄像头的画面
        img_copy = img.copy()
        '''图像预处理: 原始图像 - 灰度图像 - 图像滤波(双边/高斯) - 二值化(边缘检测/阈值化) - 形态学处理'''
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度
        filtered_img = cv2.GaussianBlur(gray_img, (5, 5), 0)  # 高斯滤波

        '''【基于图像对比度的自适应阈值(推荐)】'''
        '''
        使用均值np.mean()计算灰度图像gray所有像素亮度的平均值的原因：
            (1) 亮度标准化：均值能客观反映当前图像的整体亮度水平
                不同光照条件下的图像亮度差异很大
                明亮环境：像素值普遍接近255
                昏暗环境：像素值可能集中在0-50
            (2) 物理意义：
                低均值（暗图像）：降低阈值以捕捉微弱边缘
                高均值（亮图像）：提高阈值避免噪声干扰
        
        # avg_intensity = np.mean(gray_img)                   # 【1】传入滤波后的filtered_img来计算均值，在大多数实际场景中表现更好，尤其是存在噪声时
        avg_intensity = np.mean(filtered_img)               # 【2】传入灰度图像(传入未滤波的gray_img来计算均值，可追求极致细节且图像干净)
        low = max(50, avg_intensity * 0.5)          # 动态下限
        high = min(200, avg_intensity * 1.5)        # 动态上限(推荐比例：threshold2 ≈ 2~3 × threshold1)
        # binary_img = cv2.Canny(gray_img, low, high)         # 【1】对灰度图像作边缘检测：可以识别到细节(靶环的椭圆)
        binary_img = cv2.Canny(filtered_img, low, high)     # 【2】对滤波图像作边缘检测：丢失部分细节，但是有更好的抗干扰性'''

        '''【固定阈值(low = 100, high = 200)】'''
        '''
        Canny阈值调节方法：
            边缘断裂不连续	    降低threshold1    检测更多弱边缘
            背景噪声过多	    提高threshold2    只保留强边缘
            重要边缘丢失	    同时降低两个阈值    增加灵敏度
            边缘太粗/不精确    提高两个阈值       边缘更细更严格'''
        # 获取滑动条值
        low = cv2.getTrackbarPos('low', 'Canny Edge Demo')
        high = cv2.getTrackbarPos('high', 'Canny Edge Demo')
        binary_img = cv2.Canny(filtered_img, low, high)

        binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)  # 形态学处理：闭运算(先膨胀后腐蚀)
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找轮廓(树状结构)
        '''【测试】绘制轮廓：观察找出的所有轮廓'''
        cv2.drawContours(img_copy, contours[:], -1, (0,0,255), 1)
        cv2.putText(img_copy, f"low = {low}", (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 1)
        cv2.putText(img_copy, f"high = {high}", (0, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 1)



        '''显示图像'''
        cv2.imshow("Canny Edge Demo", img_copy)
        '''按键检测'''
        key = cv2.waitKey(1)  # 等待按键输入(1ms延迟)
        print(key)
        # 【'q'键退出 / ESC键的ASCII码】
        if key == ord('q') or key == 27:
            print("退出程序")
            cap.release()  # 释放指针(摄像头资源)
            cv2.destroyAllWindows()  # 关闭所有OpenCV窗口
            break
        # 【's'键：保存当前帧】
        elif key == ord('s'):
            cv2.imwrite('对滤波图像作边缘检测.jpg', img_copy)
