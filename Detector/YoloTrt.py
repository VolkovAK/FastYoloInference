import os
import onnx # we must import onnx before pytorch to avoid segfault while onnx converting :/
import torch
import tensorrt as trt
import cv2
import re

import yolo2onnx
import onnx2tensorrt
import processing
import cfgparser

class YOLO_TRT():
    def __init__(self,
                 cfg_file_path,
                 weights_file_path,
                 classes_file_path,
                 batch_size=8,
                 obj_thresh=0.3,
                 nms_thresh=0.5,
                 test_aug=0):
        assert batch_size > 0
        assert test_aug in [0, 1, 2, 3] # 0 - no aug, 1 - test flip, 2 - test flip or hsv adjust, 3 - test flip and hsv adjust
        self.batch_size = batch_size * (test_aug + 1)
        self.test_aug = test_aug

        weights_file_name = os.path.basename(weights_file_path)
        cfg_file_name = os.path.basename(cfg_file_path)
        if not os.path.exists(cfg_file_path):
            raise Exception('Error: {} does not exists, init failed!'.format(cfg_file_path))
        with open(cfg_file_path, 'r') as cfg_file:
            yolo_cfg = cfg_file.read()
        anchors = cfgparser.get_anchors(yolo_cfg)
        net_width = cfgparser.get_net_width(yolo_cfg)
        self.input_size = (net_width, net_width)

        if not os.path.exists(classes_file_path):
            raise Exception('Error: {} does not exists, init failed!'.format(classes_file_path))
        with open(classes_file_path, 'r') as classes_file:
            self.classes_num = len(classes_file.read().split())

        engine_file_path = os.path.join(os.path.dirname(cfg_file_name), '{}_{}_b{}.trt'.format(
            cfg_file_name.rstrip('.cfg'), 
            weights_file_name.rstrip('.weights').rstrip('.pt').rstrip('.pth'),
            batch_size))
        onnx_file_path = os.path.join(os.path.dirname(cfg_file_name), '{}_{}_b{}.onnx'.format(
            cfg_file_name.rstrip('.cfg'), 
            weights_file_name.rstrip('.weights').rstrip('.pt').rstrip('.pth'),
            batch_size))

        if not os.path.exists(engine_file_path):
            print('Engine is not found, building...')
            if not os.path.exists(onnx_file_path):
                print('Onnx is not found, converting...')
                yolo2onnx.convert(cfg_file_path, weights_file_path, onnx_file_path)
                print('ONNX {} saved'.format(onnx_file_path))
            onnx2tensorrt.build_engine(onnx_file_path, batch_size, engine_file_path)
            print(('TRT engine {} saved'.format(engine_file_path))

        TRT_LOGGER = trt.Logger()
        with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        print('Execution context created')

        self.preprocessor = processing.Pre(self.input_size)
        self.postprocessor = processing.Post(obj_thresh, nms_thresh, self.classes_num, anchors, self.batch_size, self.input_size)

        self.bindings = []
        self.outputs = []
        for binding in self.engine:
            index = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                self.context.set_binding_shape(index, (batch_size, 3,) + self.input_size)
                self.bindings.append(0) # just placeholder
            else:
                shape = self.context.get_binding_shape(index)
                # Allocate host and device buffers
                dtype_torch = self.torch_dtype_from_trt(self.engine.get_binding_dtype(index))
                torch_mem = torch.empty(size=tuple(shape), dtype=dtype_torch, device='cuda')
                self.bindings.append(int(torch_mem.data_ptr()))
                self.outputs.append(torch_mem)



    def torch_dtype_from_trt(self, dtype):
        if dtype == trt.int8:
            return torch.int8
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError('%s is not supported by torch' % dtype)


    def detect(self, images: list):
        '''
        Requires list of numpy arrays. Doing all preprocessing and postprocessing.
        Returns lists of:
        1) bounding boxes (top left, bottom right)
        2) classes for every bbox
        3) confidences
        N-th element of output list holds data for n-th input image

        '''
        images_torch = self.preprocessor.process_batch(images)
        self.bindings[0] = int(images_torch.data_ptr())

        inference_status = self.context.execute_v2(bindings=self.bindings)

        boxes, classes, confs, batch_ids = self.postprocessor.process_batch(self.outputs, [image.shape[:2] for image in images])

        out_boxes = []
        out_classes = []
        out_confs = []
        for img_n in range(len(images)):
            out_boxes.append(boxes[batch_ids == img_n])
            out_classes.append(classes[batch_ids == img_n])
            out_confs.append(confs[batch_ids == img_n])

        return out_boxes, out_classes, out_confs
