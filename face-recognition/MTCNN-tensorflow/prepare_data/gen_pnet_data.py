"""
   对初始图片进行随机简餐，按照IOU制作pnet 的分类样本，bboxes数据，写入txt文件
"""
import numpy as np
import cv2
import os
import numpy.random as npr
from prepare_data.utils import IoU

anno_file = "wider_face_train.txt"
im_dir = "WIDER_train/images"
pos_save_dir = "12/positive"
part_save_dir = "12/part"
neg_save_dir = '12/negative'
save_dir = "./12"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)

f1 = open(os.path.join(save_dir, 'pos_12.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_12.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_12.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # part
idx = 0
# 对每张图片进行随机切割
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    # 图片路径
    im_path = annotation[0]
    # boxed change to float type
    bbox = map(float, annotation[1:])
    # gtboxes --shape[N,4]
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))

    height, width, channel = img.shape

    neg_num = 0
    # 切割50张随机neg,写入txt，label为0,不需要bbox
    while neg_num < 50:
        # 随机切割后resize为(12,12)
        size = npr.randint(12, min(width, height) / 2)
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)
        # random crop
        crop_box = np.array([nx, ny, nx + size, ny + size])
        # 计算切割后的图片与所有gtboxes的iou
        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)
        # 与所有gtboxes的iou<0.3 即为neg
        if np.max(Iou) < 0.3:
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            # neg的label为0
            f2.write("12/negative/%s.jpg" % n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    for box in boxes:
        # box (x_left, y_top, x_right, y_bottom) absolut value
        x1, y1, x2, y2 = box
        # gt's width
        w = x2 - x1 + 1
        # gt's height
        h = y2 - y1 + 1

        # ignore small faces
        # in case the ground truth boxes of small faces are not accurate
        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        # 对每个gtbox 选取五个靠近的neg,写入txt，label为0,不需要bbox
        for i in range(5):
            size = npr.randint(12, min(width, height) / 2)
            # delta_x and delta_y are offsets of (x1, y1)
            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))
            # 排除超出图片范围的
            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                # Iou with all gts must below 0.3
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("12/negative/%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
        # 对每个gtbox 选取pos,part ,bbox为相对值
        for i in range(20):
            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))

            # delta here is the offset of box center
            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)
            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])
            # 转换为相对offset
            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)
            cropped_im = img[ny1: ny2, nx1: nx2, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                # pos label = 1
                f1.write("12/positive/%s.jpg" % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                # part label = -1
                f3.write("12/part/%s.jpg" % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
    idx += 1
    if idx % 100 == 0:
        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
