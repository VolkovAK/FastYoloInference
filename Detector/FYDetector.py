import numpy as np
import cv2
import os
import time

from Detector import YoloTrt

class FYDetector(): 
    '''
    FYDetector - Fast YOLO Detector
    Applies TensorRT and heavily use of PyTorch GPU calculations for best inference time.
    '''

    def __init__(self, config_path, weights_path, names_path, obj_thresh=0.3, filter_classes=None, batch_size=1):
        '''
        config_path    - str       - path to darknet cfg file (yolov3.cfg)
        weights_path   - str       - path to darknet weights file (yolov3.weights)
        names_path     - str       - path to classes names (coco.names)
        obj_thresh     - float     - confidence threshold for objects
        filter_classes - list[int] - list of classes to output
        batch_size     - int       - size of batch for batch inference, constant in time
        '''

        if not os.path.exists(config_path):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(config_path)+"`")
        if not os.path.exists(weights_path):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weights_path)+"`")

        self.yolo = YoloTrt.YOLO_TRT(config_path, weights_path, names_path, batch_size=batch_size, obj_thresh=obj_thresh)
        if type(filter_classes) is list:
            self.acceptable_classes = filter_classes


    def detect(self, frame):
        boxes, classes, confs, batch_ids = self.yolo.detect([frame])
        return boxes

    def detect_batch(self, frames):
        boxes, classes, confs, batch_ids = self.yolo.detect(frames)

        return boxes

