import cv2
import numpy as np
import time
import onnx # we must import onnx before pytorch to avoid segfault while onnx converting :/ (seems to be solved in pytorch 1.5.0)
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

    detector = FYDetector(cfg_path, weights_path, names_path, batch_size=8)

    img = cv2.imread('img.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for _ in range(30):
        prev = time.time()
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)
        #start.record()
        out = detector.detect([img] * detector.yolo.batch_size)
        torch.cuda.synchronize()
        #end.record()
        #print(start.elapsed_time(end))
        print('total: {} ms'.format((time.time()-prev)*1000))
    print('total:')
    print('preprocessing: {:.5} ms'.format(np.median(detector.yolo.timings['pre'])))
    print('ecxecution {:.5} ms'.format(np.median(detector.yolo.timings['exec'])))
    print('postprocessing: {:.5} ms'.format(np.median(detector.yolo.timings['post'])))
    img_bboxed = draw_bboxes(img, out[0])
    cv2.imwrite('output.png', img_bboxed[:,:,::-1])
    print(out)


if __name__ == '__main__':
    main()
