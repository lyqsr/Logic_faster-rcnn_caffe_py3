# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

from fast_rcnn.config import cfg  # loki

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    if cfg.IS_COLOR_IMG or cfg.IS_3C_GRAY_IMG:
        # loki # (rgb or bgr) color image or (3 channels) image
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                        dtype=np.float32)
    else:
        # loki # (1 channel) gray image
        blob = np.zeros((num_images, max_shape[0], max_shape[1], 1),
                        dtype=np.float32)
    for i in range(num_images):  # python3 # xrange
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):  # loki # lib/roi_data_layer/minibatch.py
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    if 0 == cfg.TRAIN.Logic_scale_method:
        im_scale = float(target_size) / float(im_size_min)
    elif 1 == cfg.TRAIN.Logic_scale_method:
        im_scale = float(cfg.TRAIN.Logic_scale)
    elif 2 == cfg.TRAIN.Logic_scale_method:
        im_scale = float(max_size) / float(im_size_max)
    else:
        im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    if float(1.0) != im_scale:
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        # interpolation=cv2.INTER_LINEAR)
                        interpolation=cv2.INTER_CUBIC)  # loki # cv2.resize

    if (not cfg.IS_COLOR_IMG) and (not cfg.IS_3C_GRAY_IMG):
        # loki # (1 channel) gray image
        h = im.shape[0]
        w = im.shape[1]
        im = im.reshape((h, w, 1))
    return im, im_scale
