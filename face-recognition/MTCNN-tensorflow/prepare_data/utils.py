import numpy as np
import os

def IoU(box, boxes):
    """
    Compute IoU between detect box and gt boxes
    :param box: numpy, --shape[(4)5,]: xmin,ymin,xmax,ymax,(score)
    :param boxes: numpy, -shape[N,4]: xmin,ymin,xmax,ymax, gtboxes
    :return ovr: numpy, -shape[N]
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square
    # 一张图片中的bbox_c缩放到w,h较大值为边的正方形

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()
    # +1下边全抵消了

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def getDataFromTxt(txt, dirname):
    """
    读取txt数据
    :param txt: 文档名称
    :param dirname: 图片上级目录
    :return result：list of  (img_path, bbox, landmark), img_path:图片路径，bbox:脸的bbox,landmark: numpy -shape(5,2) absolute
    """
    with open(txt, 'r') as fd:
        lines = fd.readlines()

    result = []
    for line in lines:
        line = line.strip()
        components = line.split(' ')
        img_path = os.path.join(dirname, components[0]) # file path
        # bbox:xmin,ymin,xmax,ymax
        bbox = (int(components[1]), int(components[3]), int(components[2]), int(components[4]))
        # 左眼 右眼 鼻 左嘴 右嘴 各自的(x,y)
        landmark = np.zeros((5, 2))
        for index in range(0, 5):
            rv = (float(components[5+2*index]), float(components[5+2*index+1]))
            landmark[index] = rv
        result.append((img_path, bbox, landmark))
    return result