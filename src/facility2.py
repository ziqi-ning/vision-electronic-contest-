# -*- coding:utf-8 -*- 
import RPi.GPIO as GPIO
import time
import asyncio

"""
这个外设库是控制树莓派的LED灯，扩展板是厂家画的，设备树没有配置的暂时可以不用管这个库
"""

class light:
    def __init__(self):
        self.Red=23
        self.Green=25
        self.Blue=24
        GPIO.setwarnings(False)#忽略警告
        GPIO.setmode(GPIO.BCM)#GPIO采用BCM编码方式
        GPIO.setup(self.Blue, GPIO.OUT, initial=GPIO.HIGH)#初始化为输出口且初始电平为高
        GPIO.setup(self.Red, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.Green, GPIO.OUT, initial=GPIO.HIGH)
        #IO口输出高电平为灭、低电平为亮
        GPIO.output(self.Red, 1)
        GPIO.output(self.Green, 1)
        GPIO.output(self.Blue, 1)

    async def setColor(self, color):
        if color == "indigo":
            GPIO.output(self.Red, 0)
            GPIO.output(self.Green, 1)
            GPIO.output(self.Blue, 1)
            #print("Color set to red")
        elif color == "purple":
            GPIO.output(self.Red, 1)
            GPIO.output(self.Green, 0)
            GPIO.output(self.Blue, 1)
            #print("Color set to green")
        elif color == "yellow":
            GPIO.output(self.Red, 1)
            GPIO.output(self.Green, 1)
            GPIO.output(self.Blue, 0)
            #print("Color set to blue")
        elif color == "blue":
            GPIO.output(self.Red, 0)
            GPIO.output(self.Green, 0)
            GPIO.output(self.Blue, 1)
        elif color == "green":
            GPIO.output(self.Red, 0)
            GPIO.output(self.Green, 1)
            GPIO.output(self.Blue, 0)
        elif color == "red":
            GPIO.output(self.Red, 1)
            GPIO.output(self.Green, 0)
            GPIO.output(self.Blue, 0)
        elif color == "empty":
            GPIO.output(self.Red, 0)
            GPIO.output(self.Green, 0)
            GPIO.output(self.Blue, 0)
        elif color =="white":
            GPIO.output(self.Red, 1)
            GPIO.output(self.Green, 1)
            GPIO.output(self.Blue, 1)
        else:
            print("Invalid color:{}".format(color))




    

