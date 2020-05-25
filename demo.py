import cv2
import numpy as np
import time
import torch

from Detector.FYDetector import FYDetector


def draw_bboxes(img, bboxes):
    for bbox in bboxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 5)
    return img

def main():

    cfg_path = '../yolov3.cfg'
    weights_path = '../yolov3.weights'
    names_path = '../coco.names'

    detector = FYDetector(cfg_path, weights_path, names_path)

    img = cv2.imread('img.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for _ in range(100):
        prev = time.time()
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)
        #start.record()
        out = detector.detect(img)
        #torch.cuda.synchronize()
        #end.record()
        #print(start.elapsed_time(end))
        print('total: {} ms'.format((time.time()-prev)*1000))
    img_bboxed = draw_bboxes(img, out[0])
    cv2.imwrite('output.png', img_bboxed[:,:,::-1])
    print(out)


if __name__ == '__main__':
    main()
