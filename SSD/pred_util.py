import numpy as np

def locs_decode(pred_locs_offset,anchors_layer,scaling=(0.1, 0.1, 0.2, 0.2)):
    """
    将单层预测出的位置偏移值解码回[ymin,xmin,ymax,xmax]
    :param pred_locs_offset: numpy, --shape[1,H,W,B,4], 单层预测出的位置偏移值
    :param anchors_layer: list of numpy, [y,x,h,w], 单层对应的anchors_layer
    :param scaling: 缩放比例
    :return bboxes: numpy, --shape[1,H,W,B,4], 单层预测出的位置相对值
    """
    # reshape为[H*W,B,4]
    shape = pred_locs_offset.shape
    pred_locs_offset = np.reshape(pred_locs_offset,(-1, shape[-2], shape[-1]))
    yref, xref, href, wref = anchors_layer
    # reshape为[H*W,1]
    xref = np.reshape(xref, [-1, 1])
    yref = np.reshape(yref, [-1, 1])
    # 解码
    cx = pred_locs_offset[:, :, 0] * wref * scaling[0] + xref
    cy = pred_locs_offset[:, :, 1] * href * scaling[1] + yref
    w = wref * np.exp(pred_locs_offset[:, :, 2] * scaling[2])
    h = href * np.exp(pred_locs_offset[:, :, 3] * scaling[3])
    # bboxes: [ymin, xmin, xmax, ymax]
    locs = np.zeros_like(pred_locs_offset)
    locs[:, :, 0] = cy - h / 2.
    locs[:, :, 1] = cx - w / 2.
    locs[:, :, 2] = cy + h / 2.
    locs[:, :, 3] = cx + w / 2.
    locs = np.reshape(locs, shape)
    return locs

def ssd_bboxes_select_layer(predictions_layer,
                            localizations_layer,
                            anchors_layer,
                            select_threshold=0.5,
                            decode=True):
    """
    从单个预测层中按threshold对prediction做出筛选
    :param predictions_layer: numpy, --shape[1,H,W,B,21]
    :param localizations_layer: numpy, --shape[1,H,W,B,4]
    :param anchors_layer: list of numpy, [y,x,h,w]
    :param select_threshold: prediction的筛选阈值
    :param decode: 解码
    :return classes: 1D numpy, 分类
             scores: 1D numpy, 预测概率
             bboxes: 2D numpy, 对应的locs
    """
    # 解码
    if decode:
        localizations_layer = locs_decode(localizations_layer, anchors_layer)

    p_shape = predictions_layer.shape
    batch_size = p_shape[0] if len(p_shape) == 5 else 1
    # reshape为[1,H*W*B,21]
    predictions_layer = np.reshape(predictions_layer,
                                   (batch_size, -1, p_shape[-1]))
    l_shape = localizations_layer.shape
    # reshape为[1,H*W*B,21]
    localizations_layer = np.reshape(localizations_layer,
                                     (batch_size, -1, l_shape[-1]))

    # 根据predictions> select_threshold来筛选
    sub_predictions = predictions_layer[:, :, 1:]
    idxes = np.where(sub_predictions > select_threshold)
    classes = idxes[-1]+1
    scores = sub_predictions[idxes]
    bboxes = localizations_layer[idxes[:-1]]

    return classes, scores, bboxes

def ssd_bboxes_select(predictions_net,
                      localizations_net,
                      anchors_net,
                      select_threshold=0.5,
                      decode=True):
    """
    循环筛选所有层
    :param predictions_net: list of numpy, 其中单个shape为[1,H,W,B,21]
    :param localizations_net: list of numpy, 其中单个shape为[1,H,W,B,4]
    :param anchors_net: list of list of numpy,
    :param select_threshold: 筛选prediction的阈值
    :param decode: 解码
    :return classes: 1D numpy, 所有预测层筛选出的class
             scores: 1D numpy, 对应的prediction
             bboxes: 2D numpy, 对应的locs
    """
    l_classes = []
    l_scores = []
    l_bboxes = []
    for i in range(len(predictions_net)):
        classes, scores, bboxes = ssd_bboxes_select_layer(
            predictions_net[i], localizations_net[i], anchors_net[i],
            select_threshold, decode)
        l_classes.append(classes)
        l_scores.append(scores)
        l_bboxes.append(bboxes)

    classes = np.concatenate(l_classes, 0)
    scores = np.concatenate(l_scores, 0)
    bboxes = np.concatenate(l_bboxes, 0)
    return classes, scores, bboxes

def bboxes_clip(bboxes):
    """
    对图片外的bboxes进行剪裁
    :param bboxes: 2D numpy, shape--[N,4]
    :return bboxes: 2D numpy, shape--[N,4]
    """
    bboxes = np.copy(bboxes)
    bboxes = np.transpose(bboxes)
    # 设置ymin,xmin,ymax,xmax的界限
    bboxes[0] = np.maximum(bboxes[0], 0.0)
    bboxes[1] = np.maximum(bboxes[1], 0.0)
    bboxes[2] = np.minimum(bboxes[2], 1.0)
    bboxes[3] = np.minimum(bboxes[3], 1.0)
    bboxes = np.transpose(bboxes)
    return bboxes

def bboxes_sort(classes, scores, bboxes, top_k=400):
    """
    筛选前k个bboxes, 并以prediction概率的降序输出
    """
    idxes = np.argsort(-scores)
    classes = classes[idxes][:top_k]
    scores = scores[idxes][:top_k]
    bboxes = bboxes[idxes][:top_k]
    return classes, scores, bboxes

def bboxes_nms(classes, scores, bboxes, nms_threshold=0.45):
    """
    使用nms算法
    """
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_jaccard(bboxes[i], bboxes[(i+1):])
            # 筛选iou大于0.45且同类的bboxes
            keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores[idxes], bboxes[idxes]

def bboxes_jaccard(bboxes1, bboxes2):
    """
    通过broadcast计算iou
    """
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard

