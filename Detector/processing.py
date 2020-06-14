import cv2
import numpy as np
import os
import time

import onnx
import torch
from torchvision import ops


import traceback
class cutotime():
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        torch.cuda.synchronize()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        self.start.record()
        torch.cuda.synchronize()

    def __exit__(self, exc_type, exc_value, tb):
        torch.cuda.synchronize()
        self.end.record()
        torch.cuda.synchronize()
        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)
        print('{} - {:.5f} ms'.format(self.name, self.start.elapsed_time(self.end)))

    def start(self):
        self.__enter__()
        return self

    def stop(self):
        self.__exit__(None, None, None)




class Pre(object):

    def __init__(self, yolo_input_resolution):
        """Initialize with the input resolution for YOLOv3, which will stay fixed.

        Keyword arguments:
        yolo_input_resolution -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.yolo_input_resolution = yolo_input_resolution

    def process_batch(self, input_images):
        inps = []
        for image in input_images:
            img = cv2.resize(image, self.yolo_input_resolution)
            img_t = torch.as_tensor(img)
            img_t.unsqueeze_(0)
            inps.append(img_t.to(device='cuda:0', non_blocking=True))
        images = torch.cat(inps, axis=0)
        images = images.permute([0, 3, 1, 2]).flatten()
        images = images/255.0
        return images


class Post(object):
    """Class for post-processing the three outputs tensors from YOLO"""

    def __init__(self,
                 obj_threshold,
                 nms_threshold,
                 classes_num,
                 anchors,
                 masks,
                 batch_size,
                 yolo_input_resolution):
        """Initialize with all values that will be kept when processing several frames.
        Assuming 3 outputs of the network in the case of (large) YOLOv3.

        Keyword arguments:
        yolo_masks -- a list of 3 three-dimensional tuples for the YOLO masks
        yolo_anchors -- a list of 9 two-dimensional tuples for the YOLO anchors
        object_threshold -- threshold for object coverage, float value between 0 and 1
        nms_threshold -- threshold for non-max suppression algorithm,
        float value between 0 and 1
        input_resolution_yolo -- two-dimensional tuple with the target network's (spatial)
        input resolution in HW order
        """
        self.masks = masks
        self.anchors = anchors
        self.object_threshold = obj_threshold
        self.batch_size = batch_size
        self.nms_threshold = nms_threshold
        self.classes_num = classes_num
        self.input_resolution_yolo = torch.tensor(yolo_input_resolution).to(torch.float16).cuda()

        self.grids = []
        yis = yolo_input_resolution[0] # create different for X and Y
        self.sizes = [yis//2//2//2//2//2, yis//2//2//2//2, yis//2//2//2] # create more flexible
        for size in self.sizes:
            col = np.tile(np.arange(0, size), size).reshape(-1, size)
            row = np.tile(np.arange(0, size).reshape(-1, 1), size)

            col = col.reshape(size, size, 1, 1).repeat(3, axis=-2)
            row = row.reshape(size, size, 1, 1).repeat(3, axis=-2)
            grid = np.concatenate((col, row), axis=-1)
            self.grids.append(torch.tensor(grid).flatten(end_dim=-2).to(torch.float16).cuda())

        self.sizes_cuda = [torch.tensor([size, size]).cuda() for size in self.sizes] # ???????
        self.number_two = torch.tensor(2).cuda()
        self.anchors_cuda = []
        self.image_dims = None
        for i_m, mask in enumerate(self.masks):
            anchor = torch.tensor([self.anchors[i_m][i] for i in mask]).to(torch.float16).cuda()
            anchor = anchor.repeat(self.sizes[i_m] * self.sizes[i_m], 1)
            anchor = anchor / self.input_resolution_yolo
            self.anchors_cuda.append(anchor)

        self.output_shapes_initial = [(batch_size, 
                                      (classes_num + 5),  len(self.masks[i]), 
                                      self.sizes[i], 
                                      self.sizes[i]) for i in range(len(self.sizes))]
        self.output_shapes = [(batch_size,
                               self.sizes[i],  self.sizes[i], 
                               len(self.masks[i]),
                               classes_num + 5) for i in range(len(self.sizes))]
        


    def process_batch(self, outputs, raw_sizes):
        outputs_reshaped = []
        for i, output in enumerate(outputs): # 3 scales
            output_reshaped = self._reshape_output_batch(i, output)
            outputs_reshaped.append(output_reshaped)

        boxes, categories, confidences, batch_inds = self._process_yolo_output_batch(outputs_reshaped, raw_sizes)

        #if boxes.shape[0] != 0:
        #    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        #    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        #    boxes[:, 0] += boxes[:, 2]//2
        #    boxes[:, 1] += boxes[:, 3]//2

        return boxes, categories, confidences, batch_inds

    def _reshape_output_batch(self, number, output):
        """Reshape a TensorRT output from NCHW to NHWC format (with expected C=255),
        and then return it in (batch,height,width,3,85) dimensionality after further reshaping.

        Keyword argument:
        output -- an output from a TensorRT engine after inference
        """
        #tt = cutotime('reshape')
        #tt.start()
        #print(output.data_ptr(), output.is_contiguous(), output.shape)
        output = output.reshape(self.output_shapes_initial[number]) # batch, (5 + 80), 3, h, w
        #print(output.data_ptr(), output.is_contiguous(), output.shape)
        #print(output)

        output = output.permute(0, 3, 4, 1, 2) # batch, h, w, (5+80), 3
        #print(output.data_ptr(), output.is_contiguous(), output.shape)
        output = output.reshape(self.output_shapes[number]) # batch, h * w, 3, (5 + 80)
        #print(output.data_ptr(), output.is_contiguous(), output.shape)
        #tt.stop()
        #with cutotime('flat'):
        output = output.flatten(start_dim=1, end_dim=-2)
        #print(output.data_ptr(), output.is_contiguous(), output.shape)
        #print(output)
        #print('next')
        return output 

    def _process_yolo_output_batch(self, outputs_reshaped, raw_sizes):
        """Take in a list of three reshaped YOLO outputs in (batch,height,width,3,85) shape and return
        return a list of bounding boxes for detected object together with their category and their
        confidences in separate lists.

        Keyword arguments:
        outputs_reshaped -- list of three reshaped YOLO outputs as NumPy arrays
        with shape (batch,height,width,3,85)
        resolution_raw -- (H,W)
        """
        # E.g. in YOLOv3-608, there are three output tensors, which we associate with their
        # respective masks. Then we iterate through all output-mask pairs and generate candidates
        # for bounding boxes, their corresponding category predictions and their confidences:

        boxes, categories, confidences, batch_indses = list(), list(), list(), list()
        factor = 0
        for output, mask in zip(outputs_reshaped, self.masks):
            box, category, confidence, batch_inds = self._process_feats_batch(output, mask, factor)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)
            batch_indses.append(batch_inds)
            factor += 1

        boxes = torch.cat(boxes).cpu()
        categories = torch.cat(categories).cpu()
        confidences = torch.cat(confidences).cpu()
        batch_inds = torch.cat(batch_indses).cpu()#to(device='cpu', non_blocking=True)

        # Scale boxes back to original image shape:
        for batch in batch_inds.unique():
            h, w = raw_sizes[batch]
            boxes[batch_inds == batch] = boxes[batch_inds == batch] * torch.tensor([w, h, w, h])

# CV DNN NMS_BOXES ?
        keep = ops.boxes.batched_nms(boxes, confidences, categories + (batch_inds * self.classes_num), self.nms_threshold)

        if len(keep) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])

        nms_boxes = boxes[keep].numpy()
        nms_categories = categories[keep].numpy()
        nms_scores = confidences[keep].numpy()
        nms_batches = batch_inds[keep].numpy()

        return nms_boxes, nms_categories, nms_scores, nms_batches


    def _process_feats_batch(self, output_reshaped, mask, scale_factor):
        """Take in a reshaped YOLO output in height,width,3,85 format together with its
        corresponding YOLO mask and return the detected bounding boxes, the confidence,
        and the class probability in each cell/pixel.

        Keyword arguments:
        output_reshaped -- reshaped YOLO output as NumPy arrays with shape (height,width,3,85)
        mask -- 2-dimensional tuple with mask specification for this output
        """

        #whole_proc = cutotime('whole processing').start()

        box_confidence = torch.sigmoid(output_reshaped[:, ..., 4:5]) # 4 - objectness
        #box_confidence = torch.sigmoid(output_reshaped[:, 4:5]) # 4 - objectness
        #print('box_conf',box_confidence)

        first_filter = torch.where(box_confidence >= self.object_threshold)
        #print('ff',first_filter)
        output_reshaped = output_reshaped[first_filter[:-1]]
        #print('out resh',output_reshaped)
        total_sigmoid = torch.sigmoid(output_reshaped)
        #print('ff',total_sigmoid)

        #print(torch.sigmoid(output_reshaped[:, ..., :2]).shape)
        box_xy = total_sigmoid[:, ..., :2]           # 0, 1 - x, y
        #print('xy',box_xy)
        #box_xy = torch.sigmoid(output_reshaped[:, ..., :2])           # 0, 1 - x, y
        #print(box_xy.shape)
        #print(anchors.shape)
        #print(output_reshaped.shape)
        #print(anchors[first_filter[1]])
        box_wh = torch.exp(output_reshaped[:, ..., 2:4]) * self.anchors_cuda[scale_factor][first_filter[1]]  # 2, 3 - w, h
        #print('wh',box_wh)
        #box_wh = box_wh.flatten(start_dim=0, end_dim=-2)
        #print('wh',box_wh.shape)
        #print('xy',box_xy.shape)
        #box_wh = box_wh.flatten(start_dim=1, end_dim=-2)[first_filter[:-1]]
        box_class_probs = total_sigmoid[:, ..., 5:] # 5, ... - classes probs
        #print('bcp',box_class_probs)
        #print(box_class_probs.shape)
        box_xy += self.grids[scale_factor][first_filter[1]] 
        #print('xy',box_xy)
        box_xy /= self.sizes_cuda[scale_factor]
        box_xy -= (box_wh / self.number_two)
        #print('xy',box_xy)
        boxes = torch.cat((box_xy, box_xy + box_wh), axis=-1).flatten(end_dim=-2)

        box_scores = box_confidence[first_filter[:-1]] * box_class_probs
        #print(box_scores.shape)
        #print(box_scores)
        #box_scores = box_confidence[first_filter[:-1]] * box_class_probs[first_filter[:-1]]
        box_class_scores = torch.max(box_scores, axis=-1)
        box_classes = box_class_scores.indices
        box_class_scores = box_class_scores.values
        #print(box_class_scores.shape)
        pos = torch.where(box_class_scores >= self.object_threshold)
        #print(pos)
        out = boxes[pos], box_classes[pos], box_class_scores[pos], first_filter[0][pos[0]]
        
        #whole_proc.stop()
# https://github.com/opencv/opencv/issues/17148
# scale_x_y
        return out 

