#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg, p_cfg
from fast_rcnn.test import im_detect, im_detect_loki
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# import matplotlib.pyplot as plt
import numpy as np
# import scipy.io as sio
import caffe, os, sys, cv2

import shutil
from logic_tools.pascal_voc_io import PascalVocWriter

import threading


def vis_detections(class_name, dets, thresh=0.5):
    list_obj = []
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return list_obj

    for i in inds:
        bbox = dets[i, :4]
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])
        score = dets[i, -1]
        wt = x1 - x0 + 1
        ht = y1 - y0 + 1
        list_obj.append((x0, y0, x1, y1, class_name, score, wt, ht))

    return list_obj


def create_subdir(parent, sub):
    subdir = parent + '/' + sub
    if not os.path.isdir(subdir):
        os.mkdir(subdir)
    return subdir


def demo_server(net, proj_name, np_img):
    """Detect object classes in an image using pre-computed object proposals."""
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, np_img, proj_name)

    # Visualize detections for each class
    CONF_THRESH = p_cfg[proj_name].CONF_THRESH
    NMS_THRESH = p_cfg[proj_name].NMS_THRESH
    CLASSES = p_cfg[proj_name].Logic_classes
    list_obj = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]

        # nms ^__^
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        list_t = vis_detections(cls, dets, thresh=CONF_THRESH)
        if 0 == len(list_t):
            continue

        for ele in list_t:
            list_obj.append(ele)

    Caffe_V_memory_reset(net, proj_name)
    timer.toc()
    print(('Faster Time: {:.3f}s').format(timer.diff))  # python3

    return list_obj


def demo(net, proj_name, im, image_name):
    """Detect object classes in an image using pre-computed object proposals."""
    location_no = cfg.DATA_DIR + '/' + 'location_no_obj_img'
    location_hv = cfg.DATA_DIR + '/' + 'location_hv_obj_img'
    location_anno = cfg.DATA_DIR + '/' + 'location_hv_obj_ano'

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, proj_name)

    # Visualize detections for each class
    CONF_THRESH = p_cfg[proj_name].CONF_THRESH
    NMS_THRESH = p_cfg[proj_name].NMS_THRESH
    CLASSES = p_cfg[proj_name].Logic_classes
    list_obj = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        list_t = vis_detections(cls, dets, thresh=CONF_THRESH)
        if 0 == len(list_t):
            continue
        for ele in list_t:
            list_obj.append(ele)

    timer.toc()
    print(('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0]))  # python3

    if 0 == len(list_obj):
        shutil.copy(im_file, location_no + '/' + image_name)
    else:
        shutil.copy(im_file, location_hv + '/' + image_name)

        # output xml st
        imgFileNameWithoutExt = os.path.splitext(image_name)[0]
        location_xml_filename = location_anno + '/' + imgFileNameWithoutExt + '.xml'

        height = im.shape[0]
        width = im.shape[1]
        if 3 == len(im.shape):
            channel = 3
        else:
            channel = 1

        writer = PascalVocWriter('VOC2007', image_name, [height, width, channel], localImgPath='')
        for ele in list_obj:
            xmin = int(ele[0])
            ymin = int(ele[1])
            xmax = int(ele[2])
            ymax = int(ele[3])
            name_t = ele[4]
            fen = ele[5]
            writer.addBndBox(xmin, ymin, xmax, ymax, name_t, fen)
        writer.save(targetFile = location_xml_filename)
        # output xml ed


def gen_net_body(proj_name):
    prototxt = os.path.join(cfg.MODELS_DIR, 'ZF', 'faster_rcnn_end2end', p_cfg[proj_name].proto_test_name)
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models', p_cfg[proj_name].model_name)

    if not os.path.isfile(prototxt):
        raise IOError(('{:s} not found !').format(prototxt))
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found !').format(caffemodel))

    if False:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print('\n\nLoaded network {:s}'.format(caffemodel))  # python3 # print
    return net


########################################################################################################################
# 3c image
im_ss_3c = 128 * np.ones((16, 16, 3), dtype=np.uint8)  # image small_small
im_sb_3c = 128 * np.ones((32, 32, 3), dtype=np.uint8)  # image small_big
# 1c image
im_ss_1c = 128 * np.ones((16, 16), dtype=np.uint8)  # image small_small
im_sb_1c = 128 * np.ones((32, 32), dtype=np.uint8)  # image small_big


def Caffe_V_memory_reset(net, proj_name):
    if p_cfg[proj_name].IS_COLOR_IMG or p_cfg[proj_name].IS_3C_IMG:
        im_detect_loki(net, im_ss_3c, proj_name)
        # re-malloc memory
        im_detect_loki(net, im_sb_3c, proj_name)
    else:
        im_detect_loki(net, im_ss_1c, proj_name)
        # re-malloc memory
        im_detect_loki(net, im_sb_1c, proj_name)


########################################################################################################################
NET_dict = {}
for proj_name in p_cfg:
    NET_dict[proj_name] = gen_net_body(proj_name)


########################################################################################################################
mutex_faster = threading.Lock()


def detect(np_img, proj_name='None'):
    global NET_dict

    # if not p_cfg.has_key(proj_name):  # python2
    if not (proj_name in p_cfg):  # python3
        print('do not have project name:', proj_name)
        return []

    h = np_img.shape[0]
    w = np_img.shape[1]
    if 16 > h or 16 > w:
        return []

    if mutex_faster.acquire(10):
        if False:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(cfg.GPU_ID)

        net = NET_dict[proj_name]
        list_obj = demo_server(net, proj_name, np_img)
        mutex_faster.release()
        return list_obj

###############################################################################################################
# mutex_faster_ecdhp = threading.Lock()
# def detect_e_cdhp(np_img):
#     global NET_dict
#
#     proj_name = 'e_cdhp'
#
#     #if not p_cfg.has_key(proj_name):
#     if not (proj_name in p_cfg): # python2
#         return []
#
#     h = np_img.shape[0]
#     w = np_img.shape[1]
#     if 16 > h or 16 > w:
#         return []
#
#     if mutex_faster_ecdhp.acquire(10):
#         if False:
#             caffe.set_mode_cpu()
#         else:
#             caffe.set_mode_gpu()
#             caffe.set_device(cfg.GPU_ID)
#
#         net = NET_dict[proj_name]
#         list_obj = demo_server(net, proj_name, np_img)
#         mutex_faster_ecdhp.release()
#         return list_obj


########################################################################################################################
if __name__ == '__main__':

    location_no = create_subdir(cfg.DATA_DIR, 'location_no_obj_img')
    location_hv = create_subdir(cfg.DATA_DIR, 'location_hv_obj_img')
    location_anno = create_subdir(cfg.DATA_DIR, 'location_hv_obj_ano')

    proj_name = 'yyzz_3in1'

    location = os.path.join(cfg.DATA_DIR, 'img')
    ext = '.jpg'

    im_names = []
    for root, dirs, files in os.walk(location):
        if root != location:
            break
        for file_t in files:
            if file_t.endswith(ext):
                im_names.append(file_t)

    # if not NET_dict.has_key(proj_name):  # python2
    if not (proj_name in NET_dict):  # python3
        print('do not have project name:', proj_name)
    else:
        for im_name in im_names:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', proj_name)
            print(cfg.DATA_DIR + '/' + im_name)

            # Load image
            im_file = os.path.join(cfg.DATA_DIR, 'img', im_name)
            if p_cfg[proj_name].IS_COLOR_IMG:
                np_img = cv2.imread(im_file, cv2.IMREAD_COLOR)  # loki # cv2.imread
            else:
                np_img = cv2.imread(im_file, cv2.IMREAD_GRAYSCALE)  # loki # cv2.imread
                if p_cfg[proj_name].IS_3C_GRAY_IMG:
                    np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)

            net = NET_dict[proj_name]
            demo(net, proj_name, np_img, im_name)
