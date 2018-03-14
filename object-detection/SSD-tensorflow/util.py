import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import *

def l2_normalization(inputs,scaling=True):
    """
    在channel上计算l2_norm
    :param inputs: 4D tensor, shape-[N,H,W,C]
    :param scaling: bool,是否对channel进行缩放
    :return outputs: 4D tensor, shape-[N,H,W,C]
    """
    with tf.variable_scope('L2Normalization'):
        inputs_shape = inputs.get_shape()
        channel_shape = inputs_shape[-1:]
        # 在channel上计算l2_norm
        outputs = tf.nn.l2_normalize(inputs,3, epsilon=1e-12)
        # 缩放
        if scaling:
            # scale.shape == channel.shape
            scale = slim.variable('gamma',channel_shape,tf.float32,tf.constant_initializer(1.0))
            outputs = tf.multiply(outputs, scale)

        return outputs

def cal_all_pred_layer(end_points):
    """
    计算图片经过各个特征处理层后的预测值，组成list
    :param end_points: ordereddict, 保存图片经过特征处理层前的tensor
    :return logits_pred_list: list of tensor, 对应default-boxes所有框的分类预测值，其中tensor的shape分别为[N,38,38,b,21]，[N,19,19,b,21],...,[N,1,1,b,21]
             locs_pred_list: list of tensor, 对应default-boxes所有框的位置偏移量预测值， 其中tensor的shape分别为[N,38,38,b,4]，[N,19,19,b,4],...,[N,1,1,b,4]
    """
    logits_pred_list = []
    locs_pred_list = []
    prediction_list = []
    # 分别计算对应default-box的预测值
    for i, layer in enumerate(FEAT_LAYERS):
        with tf.variable_scope(layer + '_box'):
            input = end_points[layer]
            # 第一层计算l2_norm
            if NORMALIZATOINS[i] > 0:
                input = l2_normalization(input)
            n, h, w, c = input.shape.as_list()
            num_box = len(ANCHOR_SIZES[i]) + len(ANCHOR_RATIOS[i])
            loc_pred = slim.conv2d(input, num_box*4, [3, 3], activation_fn=None,scope='conv_loc')
            # reshape为[N,H,W,B,4]
            loc_pred =tf.reshape(loc_pred, [-1, h, w, num_box, 4])
            cls_pred = slim.conv2d(input,num_box*NUM_CLASS,[3,3], activation_fn=None,scope='conv_cls')
            # reshape为[N,H,W,B,21]
            cls_pred = tf.reshape(cls_pred,[-1,h,w,num_box,NUM_CLASS])

        logits_pred_list.append(cls_pred)
        locs_pred_list.append(loc_pred)
        prediction_list.append(tf.nn.softmax(cls_pred))

    return prediction_list,logits_pred_list,locs_pred_list


def create_default_boxes():
    """
    计算所有的default-box的坐标,组成list
    :return default_boxes_loc: list of list of numpy,
    """
    default_boxes_loc = []
    # 分别计算每个特征层default-box的坐标
    for i,feat_shape in enumerate(FEAT_SHAPES):
        num_box = len(ANCHOR_SIZES[i]) + len(ANCHOR_RATIOS[i])
        # 每个像素点都形成各自的default-box
        cy,cx = np.mgrid[0:feat_shape[0],0:feat_shape[1]]
        # 以每个像素的中心点为中点，除以特征层大小得到相对位置 范围(0,1)
        cy = (cy + 0.5) * ANCHOR_STEPS[i]/IMAGE_SHAPE[0]
        cx = (cx+0.5) *ANCHOR_STEPS[i]/IMAGE_SHAPE[1]
        # cy,cx --shape[H,W,1]
        cy = np.expand_dims(cy,-1).astype('float32')
        cx = np.expand_dims(cx,-1).astype('float32')
        w = np.zeros(num_box,dtype='float32')
        h = np.zeros(num_box,dtype='float32')
        # 根据ANCHOR_SIZES,ANCHOR_RATIOS,并除以原始图片大小得到H,W的相对长度,H,W --shape[B,]
        h[0] = ANCHOR_SIZES[i][0]/IMAGE_SHAPE[0]
        w[0] = ANCHOR_SIZES[i][0]/IMAGE_SHAPE[1]
        h[1] = np.sqrt(ANCHOR_SIZES[i][0]*ANCHOR_SIZES[i][1])/IMAGE_SHAPE[0]
        w[1] = np.sqrt(ANCHOR_SIZES[i][0]*ANCHOR_SIZES[i][1])/IMAGE_SHAPE[1]
        for j,ratio in enumerate(ANCHOR_RATIOS[i]):
            h[j+2] = h[0] / np.sqrt(ratio)
            w[j+2] = w[0] * np.sqrt(ratio)
        default_boxes_loc.append([cy,cx,h,w])
    return default_boxes_loc

def convert_image_to_target(labels,bboxes,default_boxes,threshold=0.5,scaling=(0.1, 0.1, 0.2, 0.2)):
    """
    将单张图片编码成target
    :param labels: 1D Tensor(int64) 图片中的目标label
    :param bboxes: 2D Tensor(float32), --shape[num_label,4], 每个label对应的bboxes相对于图片大小的坐标 (ymin,xmin,ymax,xmax)
    :param default_boxes: list of list of numpy
    :param threshold: 筛选正样本的iou的阈值
    :param scaling: 编码缩放系数
    :return target_labels: list of tensor, 对应default-boxes所有框的分类目标值，其中tensor的shape分别为[38,38,b]，[19,19,b],...,[1,1,b]
             target_locs: list of tensor, 对应default-boxes所有框的位置偏移量目标值，其中tensor的shape分别为[38,38,b,4]，[19,19,b,4],...,[1,1,b,4]
    """
    target_labels_list = []
    target_locs_list = []
    # 分别计算相对每层default-box的值
    for default_box in default_boxes:
        # 计算四个角并且裁剪图片大小外的框
        cy,cx,h,w = default_box
        ymin = tf.maximum(cy - h / 2.0,0.0)
        xmin = tf.maximum(cx - w / 2.0,0.0)
        ymax = tf.minimum(cy + h / 2.0,1.0)
        xmax = tf.minimum(cx + w / 2.0,1.0)
        shape = (cy.shape[0],cy.shape[1],h.shape[0])
        # 每个default-box的面积
        default_area = (xmax - xmin) * (ymax - ymin)
        # 保存上一次的labels,iou,ymin等,初始化为0
        feat_labels = tf.zeros(shape,tf.int64)
        feat_iou = tf.zeros(shape)
        feat_ymin = tf.zeros(shape)
        feat_xmin = tf.zeros(shape)
        # 初始化为1,防止log0
        feat_ymax = tf.ones(shape)
        feat_xmax = tf.ones(shape)

        def iou_with_bbox(bbox):
            """
            计算交/并面积(iou)
            :param box: 单个bbox (ymin,xmin,ymax,xmax)
            :return: iou tensor, --shape[H,W,B]
            """
            # 计算交集
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_area = h * w
            # 计算并集
            union_area = default_area - inter_area + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            iou = tf.div(inter_area, union_area)
            return iou

        def condition(i, feat_labels, feat_iou,feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """
            循环条件: labels中的所有目标
            """
            r = tf.less(i, tf.shape(labels))
            return r[0]

        def body(i, feat_labels, feat_iou,feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            """
            循环体: 更新最新值
            """
            label = labels[i]
            box = bboxes[i]
            iou = iou_with_bbox(box)
            # iou>0.5，且大于之前的iou
            mask = tf.logical_and(tf.greater(iou, threshold), tf.greater(iou, feat_iou))
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, tf.float32)
            feat_labels = tf.where(mask, imask * label, feat_labels)
            feat_iou = tf.where(mask, iou, feat_iou)
            feat_ymin = tf.where(mask, fmask * box[0], feat_ymin)
            feat_xmin = tf.where(mask, fmask * box[1], feat_xmin)
            feat_ymax = tf.where(mask, fmask * box[2], feat_ymax)
            feat_xmax = tf.where(mask, fmask * box[3], feat_xmax)

            return [i + 1, feat_labels, feat_iou,
                    feat_ymin, feat_xmin, feat_ymax, feat_xmax]

        # 循环
        i = 0
        [i, feat_labels, feat_iou,feat_ymin, feat_xmin,feat_ymax, feat_xmax] = tf.while_loop(condition, body,[i, feat_labels, feat_iou,feat_ymin, feat_xmin,feat_ymax, feat_xmax])

        # 编码坐标值, 计算偏移量
        cy_offset = ((feat_ymax + feat_ymin)/2-cy)/h/scaling[0]
        cx_offset = ((feat_xmax + feat_xmin)/2-cx)/w/scaling[1]
        h_offset = tf.log((feat_ymax - feat_ymin)/h)/scaling[2]
        w_offset = tf.log((feat_xmax - feat_xmin)/w)/scaling[3]
        encode_locs = tf.stack([cx_offset,cy_offset,w_offset,h_offset],axis=-1)

        target_labels_list.append(feat_labels)
        target_locs_list.append(encode_locs)
    return target_labels_list,target_locs_list

# def locs_decode(pred_locs_offset,default_boxes,scaling=(0.1, 0.1, 0.2, 0.2)):
#     """
#     将单层预测出的位置偏移值解码回[ymin,xmin,ymax,xmax]
#     :param pred_locs_offset: numpy, --shape[1,H,W,B,4], 单层预测出的位置偏移值
#     :param default_boxes: list of numpy, [y,x,h,w], 单层对应的default-boxes
#     :param scaling: 缩放比例
#     :return bboxes: numpy, --shape[1,H,W,B,4], 单层预测出的位置相对值
#     """
#     # reshape为[H*W,B,4]
#     shape = pred_locs_offset.shape
#     pred_locs_offset = np.reshape(pred_locs_offset,(-1, shape[-2], shape[-1]))
#     yref, xref, href, wref = default_boxes
#     # reshape为[H*W,1]
#     xref = np.reshape(xref, [-1, 1])
#     yref = np.reshape(yref, [-1, 1])
#     # 解码
#     cx = pred_locs_offset[:, :, 0] * wref * scaling[0] + xref
#     cy = pred_locs_offset[:, :, 1] * href * scaling[1] + yref
#     w = wref * np.exp(pred_locs_offset[:, :, 2] * scaling[2])
#     h = href * np.exp(pred_locs_offset[:, :, 3] * scaling[3])
#     # bboxes: [ymin, xmin, xmax, ymax]
#     locs = np.zeros_like(pred_locs_offset)
#     locs[:, :, 0] = cy - h / 2.
#     locs[:, :, 1] = cx - w / 2.
#     locs[:, :, 2] = cy + h / 2.
#     locs[:, :, 3] = cx + w / 2.
#     locs = np.reshape(locs, shape)
#     return locs
#
#
#
# def ssd_bboxes_select(prediction_list,
#                       pred_locs_list,
#                       default_boxes_list,
#                       select_threshold=0.5,
#                       img_shape=(300, 300),
#                       num_classes=21,
#                       decode=True):
#     """Extract classes, scores and bounding boxes from network output layers.
#
#     Return:
#       classes, scores, bboxes: Numpy arrays...
#     """
#     l_classes = []
#     l_scores = []
#     l_bboxes = []
#     for i in range(len(prediction_list)):
#         # classes, scores, bboxes = ssd_bboxes_select_layer(
#         #     predictions_net[i], localizations_net[i], anchors_net[i],
#         #     select_threshold, img_shape, num_classes, decode)
#         if decode:
#             pred_locs = locs_decode(pred_locs_list[i], default_boxes_list[i])
#
#         # Reshape features to: Batches x N x N_labels | 4.
#         p_shape = predictions_layer.shape
#         batch_size = p_shape[0] if len(p_shape) == 5 else 1
#         predictions_layer = np.reshape(predictions_layer,
#                                        (batch_size, -1, p_shape[-1]))
#         l_shape = localizations_layer.shape
#         localizations_layer = np.reshape(localizations_layer,
#                                          (batch_size, -1, l_shape[-1]))
#
#         # Boxes selection: use threshold or score > no-label criteria.
#         if select_threshold is None or select_threshold == 0:
#             # Class prediction and scores: assign 0. to 0-class
#             classes = np.argmax(predictions_layer, axis=2)
#             scores = np.amax(predictions_layer, axis=2)
#             mask = (classes > 0)
#             classes = classes[mask]
#             scores = scores[mask]
#             bboxes = localizations_layer[mask]
#         else:
#             sub_predictions = predictions_layer[:, :, 1:]
#             idxes = np.where(sub_predictions > select_threshold)
#             classes = idxes[-1] + 1
#             scores = sub_predictions[idxes]
#             bboxes = localizations_layer[idxes[:-1]]
#
#
#
#         l_classes.append(classes)
#         l_scores.append(scores)
#         l_bboxes.append(bboxes)
#         # Debug information.
#         # l_layers.append(i)
#         # l_idxes.append((i, idxes))
#
#     classes = np.concatenate(l_classes, 0)
#     scores = np.concatenate(l_scores, 0)
#     bboxes = np.concatenate(l_bboxes, 0)
#     return classes, scores, bboxes

if __name__ == '__main__':
    default_boxes_loc = create_default_boxes()
    sess = tf.Session()

