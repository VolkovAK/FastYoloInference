import cv2
import numpy as np

from Detector.FYDetector import FYDetector


def main():

    cfg_path = '../yolov3.cfg'
    weights_path = '../yolov3.weights'
    names_path = '../coco.names'

    detector = FYDetector(cfg_path, weights_path, names_path)


if __name__ == '__main__':
    main()
