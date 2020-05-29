import cv2
import numpy as np
import os
import time

import onnx
import torch
from torchvision import ops

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
        #inps = [torch.as_tensor(cv2.resize(image, self.yolo_input_resolution).astype(np.float32), device='cuda:0').unsqueeze_(0) for image in input_images]
        #images = torch.cat(inps, axis=0)
        #images /= 255.0
        #images = images.permute([0, 3, 1, 2]).flatten()
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
            self.grids.append(torch.tensor(grid).to(torch.float16).cuda())

        self.sizes_cuda = [torch.tensor([size, size]).cuda() for size in self.sizes] # ???????
        self.number_two = torch.tensor(2).cuda()
        self.anchors_cuda = []
        self.image_dims = None
        for mask in self.masks:
            anchor = torch.tensor([self.anchors[i] for i in mask]).to(torch.float16).cuda()
            anchor = anchor.reshape([1, 1, 3, 2])  # reshape 1 1 3 2, because 3 anchors in every mask
            anchor = anchor / self.input_resolution_yolo
            self.anchors_cuda.append(anchor)

        self.output_shapes = [(batch_size, (classes_num + 5) * 3, self.sizes[0], self.sizes[0]),
                              (batch_size, (classes_num + 5) * 3, self.sizes[1], self.sizes[1]),
                              (batch_size, (classes_num + 5) * 3, self.sizes[2], self.sizes[2])]



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
        output = output.reshape(self.output_shapes[number])
        output = output.permute(0, 2, 3, 1)
        batch_size, height, width, _ = output.shape
        dim0 = batch_size
        dim1, dim2 = height, width
        dim3 = 3
        # There are classes_num=80 object categories:
        dim4 = (4 + 1 + self.classes_num)
        return torch.reshape(output, (dim0, dim1, dim2, dim3, dim4))

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
            box, category, confidence = self._process_feats_batch(output, mask, factor)
            box, category, confidence, batch_inds = self._filter_boxes_batch(box, category, confidence)
            boxes.append(box)
            categories.append(category)
            confidences.append(confidence)
            batch_indses.append(batch_inds)
            factor += 1

        boxes = torch.cat(boxes).cpu()
        categories = torch.cat(categories).cpu()
        confidences = torch.cat(confidences).cpu()
        batch_inds = torch.cat(batch_indses).cpu()

        # Scale boxes back to original image shape:
        for batch in batch_inds.unique():
            h, w = raw_sizes[batch]
            boxes[batch_inds == batch] = boxes[batch_inds == batch] * torch.tensor([w, h, w, h])

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

        anchors = self.anchors_cuda[scale_factor]

        # Reshape to N, height, width, num_anchors, box_params:
        box_xy = torch.sigmoid(output_reshaped[:, ..., :2])          # 0, 1 - x, y
        box_wh = torch.exp(output_reshaped[:, ..., 2:4]) * anchors   # 2, 3 - w, h
        box_confidence = torch.sigmoid(output_reshaped[:, ..., 4])   # 4 - objectness
        box_confidence.unsqueeze_(-1)
        box_class_probs = torch.sigmoid(output_reshaped[:, ..., 5:]) # 5, ... - classes probs

        box_xy += self.grids[scale_factor]                          
        box_xy /= self.sizes_cuda[scale_factor]
        box_xy -= (box_wh / self.number_two)
        boxes = torch.cat((box_xy, box_xy + box_wh), axis=-1)

        # boxes: centroids, box_confidence: confidence level, box_class_probs:
        return boxes, box_confidence, box_class_probs

    def _filter_boxes_batch(self, boxes, box_confidences, box_class_probs):
        """Take in the unfiltered bounding box descriptors and discard each cell
        whose score is lower than the object threshold set during class initialization.

        Keyword arguments:
        boxes -- bounding box coordinates with shape (height,width,3,4); 4 for
        x,y,height,width coordinates of the boxes
        box_confidences -- bounding box confidences with shape (height,width,3,1); 1 for as
        confidence scalar per element
        box_class_probs -- class probabilities with shape (height,width,3,classes_num)

        """
        box_scores = box_confidences * box_class_probs
        box_class_scores = torch.max(box_scores, axis=-1)
        box_classes = box_class_scores.indices
        box_class_scores = box_class_scores.values
        pos = torch.where(box_class_scores >= self.object_threshold)
        return boxes[pos], box_classes[pos], box_class_scores[pos], pos[0]

