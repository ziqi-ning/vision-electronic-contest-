import cv2
import numpy as np
import radar5
import outsite

def target_angle(img, camera_camrams):
    img, x, y = outsite.detect_ellipses(img)
    

