# 初始化串口通信模块
# 导入串口通信库和配置文件
import serial
import numpy as np


class ModeCtrl:   # 控制模式 ctr
    work_mode = 0x01
    check_show = 1


# UART缓冲区解析类定义:R
class UartBufParse:    # 接受并处理好的控制字类
    def __init__(self):
        self.uart_buf = []    # 数据缓冲区
        self._data_len = 0    # 当前数据长度
        self._data_cnt = 0    # 数据计数
        self.state = 0        # 状态机状态


class TargetCheck:    # 初始化的target类
    def __init__(self):
        self.x = 0     # x
        self.y = 0     # y
        self.pixel = 0   
        self.flag = 0    
        self.state = 0   
        self.angle = 0  # 目标角度
        self.distance = 0  # 目标距离
        self.apriltag_id = 0 # Apriltag ID
        self.img_width = 0 # 图像宽度
        self.img_height = 0 # 图像高度
        self.reserved1 = 0 # 有效标志位
        self.reserved2 = 0 # 有效标志位
        self.reserved3 = 0 # 有效标志位
        self.reserved4 = 0 # 有效标志位
        self.fps = 0 # 帧率
        self.range_sensor1 = 0 # 超声波1前
        self.range_sensor2 = 0 # 超声波2左
        self.range_sensor3 = 0 # 超声波3右
        self.range_sensor4 = 0 # 超声波4后
        self.camera_id = 0x02 # 相机ID，用来区分是否是前视还是俯视
        self.reserved1_u32 = 65536 # 保留
        self.reserved2_u32 = 105536 # 保留
        self.reserved3_u32 = 65537 # 保留
        self.reserved4_u32 = 105537 # 保留

# 串口数据解析函数
# data_buf: 接收到的数据缓冲区
# num: 数据总长度

def Receive_Anl(data_buf,num,ctr):
    # 校验和计算
    sum = 0
    i = 0
    while i<(num-1):
        sum = sum + data_buf[i]
        i = i + 1
    sum = sum%256 #取余运算
    # 校验和验证
    if sum != data_buf[num-1]:
        # print("校验失败！有数据为：{}".format(data_buf))
        return
    # 校验通过后的处理
    if data_buf[2]==0xA0:
        # 设置工作模式
        ctr.work_mode = data_buf[4]
        print("校验完毕，模式码是：",ctr.work_mode)
    

# UART数据状态机解析函数
# buf: 当前处理的字节数据
# ctr: 全局控制对象
# R: 全局数据解析对象

def uart_data_prase(R, buf, ctr):  # 处理接受数据
    # 状态机实现：检测协议头0xFF
    # print(format(buf, "X"))
    if R.state==0 and buf==0xFF:#帧头1
        R.state=1
        R.uart_buf.append(buf)
    # 状态机实现：检测协议头0xFE
    elif R.state==1 and buf==0xFE:#帧头2
        R.state=2
        R.uart_buf.append(buf)
    # 状态机实现：读取功能字（命令标识）
    elif R.state==2 and buf<0xFF:#功能字
        R.state=3
        R.uart_buf.append(buf)
    # 状态机实现：读取数据长度
    elif R.state==3 and buf<50:#数据长度小于50
        R.state=4
        R._data_len=buf  #有效数据长度
        R._data_cnt=buf+5#总数据长度
        R.uart_buf.append(buf)
    # 状态机实现：读取有效数据
    elif R.state==4 and R._data_len>0:#存储对应长度数据
        R._data_len=R._data_len-1
        R.uart_buf.append(buf)
        if R._data_len==0:
            R.state=5
    # 状态机实现：读取校验字节
    elif R.state==5:
        R.uart_buf.append(buf)
        R.state=0
        # 调用完整数据包解析函数
        # print("接收到数据包：",R.uart_buf)
        Receive_Anl(R.uart_buf,R.uart_buf[3]+5,ctr)
        R.uart_buf=[]#清空缓冲区，准备下次接收数据
    # 状态机异常处理
    else:
        R.state=0
        R.uart_buf=[]#清空缓冲区，准备下次接收数据


# 数据打包函数（用于发送）
# mode: 工作模式
# target: 目标对象（包含多种数据属性）

def package_blobs_data(mode, target):
    # 协议头定义
    HEADER = [0xFF, 0xFC]
    # 构造字节数组数据包
    data = bytearray([HEADER[0], HEADER[1], 0xA0 + mode, 0x00,
                      target.x >> 8, target.x & 0xff,
                      target.y >> 8, target.y & 0xff,
                      target.pixel >> 8, target.pixel & 0xff,
                      target.flag,
                      target.state,
                      target.angle >> 8, target.angle & 0xff,
                      target.distance >> 8, target.distance & 0xff,
                      target.apriltag_id >> 8, target.apriltag_id & 0xff,
                      target.img_width >> 8, target.img_width & 0xff,
                      target.img_height >> 8, target.img_height & 0xff,
                      target.fps,
                      target.reserved1,
                      target.reserved2,
                      target.reserved3,
                      target.reserved4,
                      target.range_sensor1 >> 8, target.range_sensor1 & 0xff,
                      target.range_sensor2 >> 8, target.range_sensor2 & 0xff,
                      target.range_sensor3 >> 8, target.range_sensor3 & 0xff,
                      target.range_sensor4 >> 8, target.range_sensor4 & 0xff,
                      target.camera_id,
                      (target.reserved1_u32 >> 24) & 0xff,
                      (target.reserved1_u32 >> 16) & 0xff,
                      (target.reserved1_u32 >> 8) & 0xff,
                      target.reserved1_u32 & 0xff,
                      (target.reserved2_u32 >> 24) & 0xff,
                      (target.reserved2_u32 >> 16) & 0xff,
                      (target.reserved2_u32 >> 8) & 0xff,
                      target.reserved2_u32 & 0xff,
                      (target.reserved3_u32 >> 24) & 0xff,
                      (target.reserved3_u32 >> 16) & 0xff,
                      (target.reserved3_u32 >> 8) & 0xff,
                      target.reserved3_u32 & 0xff,
                      (target.reserved4_u32 >> 24) & 0xff,
                      (target.reserved4_u32 >> 16) & 0xff,
                      (target.reserved4_u32 >> 8) & 0xff,
                      target.reserved4_u32 & 0xff,
                      0x00])
    # 计算数据包长度并更新长度字段
    data_len = len(data)
    data[3] = data_len - 5
    # 计算校验和（除最后一位外的所有字节和）
    checksum = sum(data[:-1]) & 0xff
    # 设置校验位
    data[-1] = checksum
    return data

def uart_data_read(R, ser, ctr):
    buf_len=ser.in_waiting
    if buf_len>0:
        buf=ser.read(buf_len)
        print("serial buf :",buf)
        for i in range(0,buf_len):
            uart_data_prase(R, buf[i], ctr)
