import cv2
import numpy as np
import imutils
from imutils import contours

# 颜色阈值
lower = np.array([0, 96, 126])
upper = np.array([97, 225, 255])
# 内核
kernel = np.ones((5, 5), np.uint8)
# 打开摄像头
vc = cv2.VideoCapture(0)
if vc.isOpened():
    flag, frame = vc.read()
    # 翻转图像
    # 这一步可以忽略，博主的摄像头是反着的
    # 所以加上这句话可以让摄像头的图像正过来
    frame = imutils.rotate(frame, 180)
    cv2.imshow("frame", frame)
else:
    flag = False
while flag:
    flag, frame = vc.read()
    # 翻转图像
    frame = imutils.rotate(frame, 180)
    draw_frame = frame.copy()
    if frame is None:
        break 
    if flag is True:
        '''下面对摄像头读取到的图像进行处理，这个步骤是比较重要的'''
        # 转换颜色空间HSV
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 颜色识别
        img = cv2.inRange(frame_hsv, lower, upper)
        # 膨胀操作
        dilation = cv2.dilate(img, kernel, iterations=1)
        # 闭操作
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
        # 高斯滤波
        closing = cv2.GaussianBlur(closing, (5, 5), 0)
        # 边缘检测
        edges = cv2.Canny(closing, 10, 20)
        '''上面进行那么多操作就是为了得到更好的目标图形，具体效果因环境而异'''
        # 寻找轮廓
        cnts, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # 判断轮廓数量也就是判断是否寻找到轮廓，如果没有找到轮廓就不继续进行操作
        if len(cnts) > 0:
            # 存放轮廓面积的列表
            s = []
            # 存放最大轮廓的索引
            max_index = 0
            # 获得排序后的轮廓列表以及每个轮廓对应的外接矩形
            (cnts, boundingRects) = contours.sort_contours(cnts)
            # 寻找面积最大的轮廓的索引
            for cnt in cnts:
                s.append(cv2.contourArea(cnt))
            max_index = s.index(max(s))
            # 根据面积最大轮廓的索引找到它的外接矩形的信息
            (x, y, w, h) = boundingRects[max_index]
            # 画矩形
            frame_out = cv2.rectangle(
                         draw_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("frame", draw_frame)
        if cv2.waitKey(10) == 27:
            break
vc.release()
cv2.destroyAllWindows()