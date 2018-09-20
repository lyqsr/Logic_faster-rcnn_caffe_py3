# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

import platform
platform_str = platform.python_version()
Python_Main_Version = platform_str[0]

########################################################################################################################
SNAPSHOT_ITERS = 20000  # loki # only for training

Logic_scale = 1.0  # loki # only for '1 == Logic_scale_method'

LOKI_SCALES = 1600
LOKI_MAX_SIZE = 1600

# 0, py-faster-rcnn default
# 1, use 'Logic_scale'
# 2, use the max w/h
Logic_scale_method = 2

# 0, do not use Hingeloss
# 1, use Hingeloss
Logic_IS_Hingeloss = 0


########################################################################################################################
# '0 == PROJECT_ID' or else
labels_faster_rcnn_original = ('__background__',  # always index 0
                                 'aeroplane',
                                 'bicycle',
                                 'bird',
                                 'boat',
                                 'bottle',
                                 'bus',
                                 'car',
                                 'cat',
                                 'chair',
                                 'cow',
                                 'diningtable',
                                 'dog',
                                 'horse',
                                 'motorbike',
                                 'person',
                                 'pottedplant',
                                 'sheep',
                                 'sofa',
                                 'train',
                                 'tvmonitor',)


########################################################################################################################
# '1 == PROJECT_ID'
labels_yyzz_3in1 = ('__background__',  # always index 0
                    'emblem', 'emblem_90', 'emblem_180', 'emblem_270',  # yyzz
                    'uni_credit_no',
                    'paper_no',
                    'paper_s_no',
                    'reg_no',
                    'org_no',
                    'tax_reg_no',
                    'social_reg_no',
                    'statistics_reg_no',
                    'company_name',
                    'company_type',
                    'company_address',
                    'legal_person',
                    'reg_money',
                    'establish_date',
                    'business_term',
                    'business_scope',
                    'dealer_name',
                    'deal_address',
                    'deal_type',
                    'reg_date',
                    'company_s_type',
                    'qr_code',
                    'taxpayer_id',  # 24 + 3

                    'title', 'title_90', 'title_180', 'title_270',  # khxk
                    'approval_no',
                    'id',
                    'first_line_left',
                    'first_line_right',
                    'second_line_left',
                    'third_line_left',
                    'third_line_right',
                    'fourth_line_left',  # 9 + 3

                    'buyer', 'buyer_90', 'buyer_180', 'buyer_270',  # zzsfp
                    'password',
                    'total_price',
                    'seller',
                    'remark',  # 5 + 3
                    )  # LOGIC ( (24 + 3) + (9 + 3) + (5 + 3) == num == 47 )


########################################################################################################################
# '2 == PROJECT_ID'
labels_e_cdhp = ('__background__',         # always index 0
                    'acceptor_guarantee',  # 1
                    'acceptor_info',       # 2
                    'borrow_guarantee',    # 3
                    'company_info_3',      # 4
                    'company_info_4',      # 5
                    'evaluate_info',       # 6
                    'title_cdhp',          # 7
                    'back_info',           # 8
                    )  # LOGIC (8 == num)


########################################################################################################################
# '3 == PROJECT_ID'
labels_yyzz = ('__background__',    # always index 0
                'uni_credit_no',    # 1
                'reg_no',           # 2
                'company_name',     # 3
                'company_type',     # 4
                'company_address',  # 5
                'legal_person',     # 6
                'reg_money',        # 7
                'establish_date',   # 8
                'business_term',    # 9
                'dealer_name',      # 10
                'deal_address',     # 11
                'deal_type',        # 12
                'reg_date',         # 13
                'company_s_type',   # 14
                'business_scope',   # 15
                'qr_code',          # 16
                'ying_ye',          # 17
                'zhi_zhao',         # 18
                'ying_ye_90',       # 19
                'ying_ye_180',      # 20
                'ying_ye_270',      # 21
                'zhi_zhao_90',      # 22
                'zhi_zhao_180',     # 23
                'zhi_zhao_270',     # 24
                )  # LOGIC (24 == num)


########################################################################################################################
PROJECT_ID = 0
if 0 == PROJECT_ID:
    Logic_classes = labels_faster_rcnn_original
elif 1 == PROJECT_ID:
    Logic_classes = labels_yyzz_3in1
elif 2 == PROJECT_ID:
    Logic_classes = labels_e_cdhp
elif 3 == PROJECT_ID:
    Logic_classes = labels_yyzz
else:
    Logic_classes = labels_faster_rcnn_original
########################################################################################################################

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#

__C.TRAIN = edict()

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
# __C.TRAIN.SCALES = (600,)
__C.TRAIN.SCALES = (LOKI_SCALES,)  # loki

# Max pixel size of the longest side of a scaled input image
# __C.TRAIN.MAX_SIZE = 1000
__C.TRAIN.MAX_SIZE = LOKI_MAX_SIZE  # loki

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2  # loki # para # (2)

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128  # loki # para # (128)

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25  # loki # para # (0.25)

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5  # loki # para # (0.5)

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5  # loki # para # (0.5)
__C.TRAIN.BG_THRESH_LO = 0.1  # loki # para # (0.1)

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True  # loki # para # (True)

# Train bounding-box regressors
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Iterations between snapshots
# __C.TRAIN.SNAPSHOT_ITERS = 10000
__C.TRAIN.SNAPSHOT_ITERS = SNAPSHOT_ITERS  # loki

# solver.prototxt specifies the snapshot path prefix, this adds an optional
# infix to yield the path: <prefix>[_<infix>]_iters_XYZ.caffemodel
__C.TRAIN.SNAPSHOT_INFIX = ''

# Use a prefetch thread in roi_data_layer.layer
# So far I haven't found this useful; likely more engineering work is required
__C.TRAIN.USE_PREFETCH = False

# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)

# Train using these proposals
__C.TRAIN.PROPOSAL_METHOD = 'selective_search'

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# Use RPN to detect objects
__C.TRAIN.HAS_RPN = False
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7  # loki # para # (0.7)
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3  # loki # para # (0.3)
# If an anchor statisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5  # loki # para # (0.5)
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256  # loki # para # (256)
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7  # loki # para # (0.7)
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000  # loki # para # (12000)
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000  # loki # para # (2000)
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16  # loki # para # (16)
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

__C.TRAIN.Logic_scale        = Logic_scale         # loki
__C.TRAIN.Logic_classes      = Logic_classes       # loki
__C.TRAIN.Logic_scale_method = Logic_scale_method  # loki
__C.TRAIN.Logic_IS_Hingeloss = Logic_IS_Hingeloss  # loki

#
# Testing options
#

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
# __C.TEST.SCALES = (600,)
__C.TEST.SCALES = (LOKI_SCALES,)  # loki # para # (600)

# Max pixel size of the longest side of a scaled input image
# __C.TEST.MAX_SIZE = 1000
__C.TEST.MAX_SIZE = LOKI_MAX_SIZE  # loki # para # (1000)

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = False

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'selective_search'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7  # loki # para # (0.7) # train (0.7)
## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000  # loki # para # (6000) # train (12000)
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300  # loki # para # (300) # train (2000)
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16  # loki # para # (16)

__C.TEST.Logic_scale        = Logic_scale          # loki
__C.TEST.Logic_classes      = Logic_classes       # loki
__C.TEST.Logic_scale_method = Logic_scale_method  # loki
__C.TEST.Logic_IS_Hingeloss = Logic_IS_Hingeloss  # loki

#
# MISC
#

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1./16.  # python3 div

# True means (3 channels) rgb or bgr color image
# False means (3 channels) or (1 channel) gray image
__C.IS_COLOR_IMG = False  # loki
# True means (3 channels) gray image
# False means (1 channel) gray image
__C.IS_3C_IMG = True  # loki

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
# __C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
if __C.IS_COLOR_IMG:
    # loki # for (rgb or bgr) color image
    __C.PIXEL_MEANS = np.array([[[127.500000, 127.500000, 127.500000]]])
else:
    if __C.IS_3C_IMG:
        # loki # for (3 channels) gray image
        __C.PIXEL_MEANS = np.array([[[127.500000, 127.500000, 127.500000]]])
    else:
        # loki # for (1 channel) gray image
        __C.PIXEL_MEANS = 127.500000

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# Model directory
__C.MODELS_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'models', 'pascal_voc'))

# Name (or path to) the matlab executable
__C.MATLAB = 'matlab'

# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default GPU device id
__C.GPU_ID = 0


def get_output_dir(imdb, net=None):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if net is not None:
        outdir = osp.join(outdir, net.name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    if '3' == Python_Main_Version:
        for k, v in a.items():  # python3 # iter
            # a must specify keys that are in b
            if k in b:  # python3 # dict
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                    'for config key: {}').format(type(b[k]),
                                                                type(v), k))

            # recursively merge dicts
            if type(v) is edict:
                try:
                    _merge_a_into_b(a[k], b[k])
                except:
                    print('Error under config key: {}'.format(k))  # python3 # print
                    raise
            else:
                b[k] = v
    else:
        for k, v in a.iteritems():  # python2 # iter
            # a must specify keys that are in b
            if not b.has_key(k):  # python2 # dict
                raise KeyError('{} is not a valid config key'.format(k))

            # the types must match, too
            old_type = type(b[k])
            if old_type is not type(v):
                if isinstance(b[k], np.ndarray):
                    v = np.array(v, dtype=b[k].dtype)
                else:
                    raise ValueError(('Type mismatch ({} vs. {}) '
                                      'for config key: {}').format(type(b[k]),
                                                                   type(v), k))

            # recursively merge dicts
            if type(v) is edict:
                try:
                    _merge_a_into_b(a[k], b[k])
                except:
                    print('Error under config key: {}'.format(k))
                    raise
            else:
                b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        if '3' == Python_Main_Version:
            for subkey in key_list[:-1]:
                assert (subkey in d)  # python3 # dict
                d = d[subkey]
            subkey = key_list[-1]
            assert (subkey in d)  # python3 # dict
        else:
            for subkey in key_list[:-1]:
                assert d.has_key(subkey)  # python2 # dict
                d = d[subkey]
            subkey = key_list[-1]
            assert d.has_key(subkey)  # python2 # dict
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value


########################################################################################################################
########################################################################################################################
########################################################################################################################


def gen_cfg(proj_name, labels,
            model_name, proto_test_name,
            Logic_scale_method, my_scale,
            my_SCALES, my_MAX_SIZE,
            Logic_IS_Hingeloss,
            IS_COLOR_IMG, IS_3C_IMG,
            CONF_THRESH, NMS_THRESH):

    one_cfg = edict()
    one_cfg.proj_name = proj_name
    one_cfg.Logic_classes = labels
    one_cfg.model_name = model_name
    one_cfg.proto_test_name = proto_test_name

    # 0, py-faster-rcnn default scale method
    # 1, loki my_scale method 1
    # 2, loki method 2
    one_cfg.Logic_scale_method = Logic_scale_method
    # loki, only for '1 == Logic_scale_method'
    one_cfg.Logic_scale = my_scale

    # 0, do not use Hingeloss
    # 1, use Hingeloss
    one_cfg.Logic_IS_Hingeloss = Logic_IS_Hingeloss

    # Scales to use during testing (can list multiple scales)
    # Each scale is the pixel size of an image's shortest side
    # __C.TEST.SCALES = (600,)
    one_cfg.SCALES = (my_SCALES,)  # LOGIC

    # Max pixel size of the longest side of a scaled input image
    # __C.TEST.MAX_SIZE = 1000
    one_cfg.MAX_SIZE = my_MAX_SIZE  # LOGIC

    # Overlap threshold used for non-maximum suppression (suppress boxes with
    # IoU >= this threshold)
    one_cfg.NMS = 0.3

    # Experimental: treat the (K+1) units in the cls_score layer as linear
    # predictors (trained, eg, with one-vs-rest SVMs).
    one_cfg.SVM = False  # ?

    # Test using bounding-box regressors
    one_cfg.BBOX_REG = True  # ?

    # Propose boxes
    one_cfg.HAS_RPN = True  # (False), (True)yml  # ?

    # Test using these proposals
    one_cfg.PROPOSAL_METHOD = 'selective_search'

    ## NMS threshold used on RPN proposals
    one_cfg.RPN_NMS_THRESH = 0.7  # (0.7) LOGIC ?
    ## Number of top scoring boxes to keep before apply NMS to RPN proposals
    one_cfg.RPN_PRE_NMS_TOP_N = 6000  # (6000) # train 12000 # LOGIC ?
    ## Number of top scoring boxes to keep after applying NMS to RPN proposals
    one_cfg.RPN_POST_NMS_TOP_N = 300  # (300) # train 2000 # LOGIC ?
    # Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
    one_cfg.RPN_MIN_SIZE = 4  ###16 # LOGIC ??? # !!!!!!

    # True means (3 channels) rgb or bgr color image
    # False means (3 channels) or (1 channel) gray image
    one_cfg.IS_COLOR_IMG = IS_COLOR_IMG  # loki
    # True means (3 channels) gray image
    # False means (1 channel) gray image
    one_cfg.IS_3C_GRAY_IMG = IS_3C_IMG  # loki

    # Pixel mean values (BGR order) as a (1, 1, 3) array
    # We use the same pixel mean for all networks even though it's not exactly what
    # they were trained with
    if one_cfg.IS_COLOR_IMG:
        one_cfg.PIXEL_MEANS = np.array([[[127.5000, 127.5000, 127.5000]]])  # loki for (rgb or bgr) color
    else:
        if one_cfg.IS_3C_IMG:
            one_cfg.PIXEL_MEANS = np.array([[[127.5000, 127.5000, 127.5000]]])  # loki for (3 channels) gray image
        else:
            one_cfg.PIXEL_MEANS = 127.5000  # loki for (1 channel) gray image

    one_cfg.CONF_THRESH = CONF_THRESH
    one_cfg.NMS_THRESH = NMS_THRESH

    return one_cfg


########################################################################################################################
G_IS_INFERENCE_G = True
p_cfg = {}


########################################################################################################################
# if G_IS_INFERENCE_G:
#     tmp_proj_name = 'e_cdhp'
#     tmp_cfg = gen_cfg(tmp_proj_name, labels_e_cdhp,
#                       'e_cdhp_zf_faster_rcnn_iter_300000.caffemodel', 'test_e_cdhp.prototxt',
#                       1, 1.5,
#                       1200, 1200,
#                       0,
#                       False, True,
#                       0.9, 0.3)
#     p_cfg[tmp_proj_name] = tmp_cfg


########################################################################################################################
# if G_IS_INFERENCE_G:
#     tmp_proj_name = 'yyzz_3in1'
#     tmp_cfg = gen_cfg(tmp_proj_name, labels_yyzz_3in1,
#                       'yyzz_3in1_zf_faster_rcnn_iter_200000_merge_50w.caffemodel', 'test_yyzz_3in1.prototxt',
#                       2, 1.0,
#                       2000, 2000,
#                       1,
#                       False, True,
#                       1.0, 0.3)
#     p_cfg[tmp_proj_name] = tmp_cfg


########################################################################################################################
if G_IS_INFERENCE_G:
    tmp_proj_name = 'yyzz'
    tmp_cfg = gen_cfg(tmp_proj_name, labels_yyzz,
                      'yyzz_lo_zf_faster_rcnn_iter_400000.caffemodel', 'test_yyzz_lo.prototxt',
                      2, 1.0,
                      1600, 1600,
                      0,
                      False, True,
                      0.80, 0.3)
    p_cfg[tmp_proj_name] = tmp_cfg


########################################################################################################################
