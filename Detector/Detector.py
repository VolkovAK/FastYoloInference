import numpy as np
import cv2
import sys

import node
import tools

import ComputerVision_pb2 as omge_cv

from ctypes import *
import math
import random
import os
import time
import darknet

import YoloTrt

class Detector():

    def __init__(self, batch_size=1):

        configPath = "/opt/models/Detector/yolo_person.cfg"
        weightPath = "/opt/models/Detector/yolo_person.weights"
        namesPath = "/opt/models/Detector/person.names"

        if not os.path.exists(configPath):
            raise ValueError("Invalid config path `" +
                            os.path.abspath(configPath)+"`")
        if not os.path.exists(weightPath):
            raise ValueError("Invalid weight path `" +
                            os.path.abspath(weightPath)+"`")

        self.yolo = YoloTrt.YOLO_TRT(configPath, weightPath, namesPath, batch_size=batch_size, obj_thresh=self.cfg['thresh'])
        self.acceptable_classes = [0]


    def detect(self, frame):
        boxes, classes, confs, batch_ids = self.yolo.detect([frame])
        return boxes

    def detect_batch(self, frames):
        boxes, classes, confs, batch_ids = self.yolo.detect(frames)

        return boxes

