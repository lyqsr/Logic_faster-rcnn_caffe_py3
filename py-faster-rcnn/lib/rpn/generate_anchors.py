# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(proj_name='faster-rcnn-original', base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    if 'faster-rcnn-original' == proj_name:
        base_anchor = np.array([1, 1, base_size, base_size]) - 1
        ratio_anchors = _ratio_enum(base_anchor, ratios)
        anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                             for i in range(ratio_anchors.shape[0])])  # python3 # xrange
    elif 'e_cdhp' == proj_name:
        anchors = [[-79., -22.,  94.,  37.],
                   [-59., -22.,  74.,  37.],
                   [-88., -52., 103.,  67.],
                   [-65.,  -7.,  80.,  22.],
                   [-52., -52.,  67.,  67.],
                   [-37., -37.,  52.,  52.]]
    elif 'yyzz_3in1' == proj_name:
        anchors = [[-172.,  -16.,  187.,   31.],
                   [-352.,  -40.,  367.,   55.],
                   [-712.,  -88.,  727.,  103.],
                   [-120.,  -24.,  135.,   39.],
                   [-248.,  -56.,  263.,   71.],
                   [-504., -120.,  519.,  135.],
                   [-84.,   -40.,   99.,   55.],
                   [-176.,  -88.,  191.,  103.],
                   [-360., -184.,  375.,  199.],
                   [-56.,   -56.,   71.,   71.],
                   [-120., -120.,  135.,  135.],
                   [-248., -248.,  263.,  263.],
                   [-36.,   -80.,   51.,   95.],
                   [-80.,  -168.,   95.,  183.],
                   [-168., -344.,  183.,  359.],
                   [-24.,  -120.,   39.,  135.],
                   [-56.,  -248.,   71.,  263.],
                   [-120., -504.,  135.,  519.],
                   [-16.,  -184.,   31.,  199.],
                   [-40.,  -376.,   55.,  391.],
                   [-88.,  -760.,  103.,  775.]]
    elif 'yyzz' == proj_name:
        anchors = [[-120., -120., 135., 135.],
                   [-120.,  -56., 135.,  71.],
                   [-120.,   -8., 135.,  23.],
                   [-56.,  -120.,  71., 135.],
                   [-56.,   -56.,  71.,  71.],
                   [-56.,   -24.,  71.,  39.],
                   [-56.,    -8.,  71.,  23.],
                   [-56.,     0.,  71.,  15.],
                   [-24.,   -56.,  39.,  71.],
                   [-24.,   -24.,  39.,  39.],
                   [-24.,    -8.,  39.,  23.],
                   [-24.,     4.,  39.,  11.],
                   [-8.,   -120.,  23., 135.],
                   [-8.,    -56.,  23.,  71.],
                   [-8.,    -24.,  23.,  39.],
                   [0.,     -56.,  15.,  71.],
                   [4.,     -24.,  11.,  39.]]
    else:
        anchors = [[]]
        print('proj_name error !', proj_name)

    anchors = np.array(anchors)
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios  # python3 div
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print (time.time() - t)  # python3 # print
    print (a)  # python3 # print
    from IPython import embed; embed()
