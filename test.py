import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from tkinter import *
import numpy as np
import imutils
from imutils import contours

import serial as ser
import struct

import pyrealsense2 as rs

def nothing(x):
    pass


pc = rs.pointcloud()
points = rs.points()

pipeline = rs.pipeline()  # 创建一个管道
config = rs.config()  # Create a config并配置要流​​式传输的管道。
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
# 使用选定的流参数显式启用设备流

# Start streaming 开启流
pipe_profile = pipeline.start(config)

# Create an align object 创建对其流对象
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
# (对其流)
align_to = rs.stream.color
align = rs.align(align_to)  # 设置为其他类型的流,意思是我们允许深度流与其他流对齐
#print(type(align))
cap = cv2.VideoCapture(0)

def led_practice(x,y):

    frames = pipeline.wait_for_frames()  # 等待开启通道

    aligned_frames = align.process(frames)  # 将深度框和颜色框对齐
    depth_frame = aligned_frames.get_depth_frame()  # ?获得对齐后的帧数深度数据(图)
    color_frame = aligned_frames.get_color_frame()  # ?获得对齐后的帧数颜色数据(图)
    img_color = np.asanyarray(color_frame.get_data())  # 把图像像素转化为数组
    img_depth = np.asanyarray(depth_frame.get_data())  # 把图像像素转化为数组

    #cv2.putText(img_color, "Distance/cm:"+str(img_depth[300, 250]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
    #cv2.putText(img_color, "Distance/cm:"+str(img_depth[x_axis, y_axis]), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, [255, 0, 255])
    #cv2.putText(img_color, "X:"+str(np.float(vtx[i][0])), (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
    #cv2.putText(img_color, "Y:"+str(np.float(vtx[i][1])), (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
    #cv2.putText(img_color, "Z:"+str(np.float(vtx[i][2])), (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 255])
    print('Distance: ',img_depth[x,y]/10)
    #cv2.imshow('depth_frame', img_color)
    #cv2.imshow("dasdsadsa", img_depth)


def ceshijian():
    # se=ser.Serial("/dev/ttyTHS1",9600,timeout=1)
    torch.cuda.current_device()
    torch.cuda._initialized = True
    def detect(save_img=False):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        #vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, 640, 480), device=device)  # init img
        #_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for path, img, im0s, im0sdepth in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, im0depth = Path(path[i]), '%g: ' % i, im0s[i].copy(),im0sdepth[i].copy()
                    #p, s = Path(path[i]), '%g: ' % i
                    #im0= np.asanyarray(color_frame.get_data())  # 把图像像素转化为数组
                else:
                    p, s,im0,im0depth = Path(path), '',im0s,im0sdepth
                    #p, s = Path(path), ''
                    #im0= np.asanyarray(color_frame.get_data())  # 把图像像素转化为数组
                #save_path = str(save_dir / p.name)
                #txt_path = str(save_dir / 'labels' / p.stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            #with open(txt_path + '.txt', 'a') as f:
                                #f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            print((torch.tensor(xyxy)[3]-torch.tensor(xyxy)[1])/(torch.tensor(xyxy)[2]-torch.tensor(xyxy)[0]))#打印箭的斜率
                            cord=torch.tensor(xyxy).numpy().tolist()
                            #print(cord[0])
                            xcord=int((cord[2]+cord[0])/2)
                            ycord=int((cord[3]+cord[1])/2)
                            cv2.circle(im0, (xcord, ycord), 8, [255, 0, 255], thickness=-1)
                            if (names[int(cls)]=='arrow'):
                                #if xcord>640:     
                                    #xcord=640
                                #if ycord>480:     
                                    #ycord=480    
                                print(im0depth[ycord,xcord]/10)
                                #se.write(str((torch.tensor(xyxy)[3]-torch.tensor(xyxy)[1])/(torch.tensor(xyxy)[2]-torch.tensor(xyxy)[0])).encode("GB2312"))
                # Print time (inference + NMS) 
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        cv2.destroyAllWindows()
                        raise StopIteration

                # Save results (image with detections)
                #if save_img:
                    #if dataset.mode == 'images':
                        #cv2.imwrite(save_path, im0)
                    #else:
                        #if vid_path != save_path:  # new video
                            #vid_path = save_path
                            #if isinstance(vid_writer, cv2.VideoWriter):
                               # vid_writer.release()  # release previous video writer

                            #fourcc = 'mp4v'  # output video codec
                            #fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            #vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                        #vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print('Done. (%.3fs)' % (time.time() - t0))

    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='F:/Program Files (x86)/yolo v5/runs/train/exp2/weights/best.pt', help='model.pt path(s)')
        parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.25, help='IOU threshold for NMS')
        parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        opt = parser.parse_args()
        print(opt)

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()

def ceshitong():
    lower = np.array([0, 96, 126])
    upper = np.array([97, 225, 255])
    # 内核
    kernel = np.ones((5, 5), np.uint8)
    # 打开摄像头
    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        flag, frame = vc.read()
        # 翻转图像
        cv2.imshow("frame", frame)
    else:
        flag = False
    while flag:
        flag, frame = vc.read()
        # 翻转图像
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
            if cv2.waitKey(10) == ord('q'):
                break
    vc.release()
    cv2.destroyAllWindows()
GUI=Tk()
GUI.title('机器人控制台')
GUI.geometry('600x400')
Button1=Button(GUI,text='开始识别箭',command=ceshijian)
Button2=Button(GUI,text='开始识别桶',command=ceshitong)
Button1.pack()
Button2.pack()
GUI.mainloop()