# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from pascal_voc_io import PascalVocWriter
import os.path
import sys

class LabelFileError(Exception):
    pass

class LabelFile(object):
    # It might be changed as window creates
    suffix = '.lif'

    def __init__(self, filename=None):
        self.shapes = ()
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)

    def savePascalVocFormat(self, filename, shapes, imagePath, imageData,
            lineColor=None, fillColor=None, databaseSrc=None):
        imgFolderPath = os.path.dirname(imagePath)
        imgFolderName = os.path.split(imgFolderPath)[-1]
        imgFileName = os.path.basename(imagePath)
        #imgFileNameWithoutExt = os.path.splitext(imgFileName)[0]
        imgFileNameWithoutExt = imgFileName
        # Read from file path because self.imageData might be empty if saving to
        # Pascal format
        image = QImage()
        image.load(imagePath)
        imageShape = [image.height(), image.width(), 1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(imgFolderName, imgFileNameWithoutExt,\
                                 imageShape, localImgPath=imagePath)
        bSave = False
        # LOGIC st
        bkg_im_w = image.width()
        bkg_im_h = image.height()
        bkg_im_w_2 = bkg_im_w - 2
        bkg_im_h_2 = bkg_im_h - 2
        # LOGIC ed
        for shape in shapes:
            points = shape['points']
            label = shape['label']
            bndbox = LabelFile.convertPoints2BndBox(points)

            bnd_0 = bndbox[0]
            bnd_1 = bndbox[1]
            bnd_2 = bndbox[2]
            bnd_3 = bndbox[3]

            if 2 > bnd_0:
                bnd_0 = 2
            if 2 > bnd_1:
                bnd_1 = 2
            if bkg_im_w_2 < bnd_2:
                bnd_2 = bkg_im_w_2
            if bkg_im_h_2 < bnd_3:
                bnd_3 = bkg_im_h_2

            if bnd_0 >= bnd_2:
                continue
            if bnd_1 > bnd_3:
                continue

            writer.addBndBox(bnd_0, bnd_1, bnd_2, bnd_3, label)
            bSave = True
            # LOGIC ed

        if bSave:
            writer.save(targetFile = filename)
        return

    @staticmethod
    def isLabelFile(filename):
        fileSuffix = os.path.splitext(filename)[1].lower()
        return fileSuffix == LabelFile.suffix

    @staticmethod
    def convertPoints2BndBox(points):
        xmin = float('inf')
        ymin = float('inf')
        xmax = float('-inf')
        ymax = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            xmin = min(x,xmin)
            ymin = min(y,ymin)
            xmax = max(x,xmax)
            ymax = max(y,ymax)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if (xmin < 1):
            xmin = 1

        if (ymin < 1):
            ymin = 1

        return (int(xmin), int(ymin), int(xmax), int(ymax))
