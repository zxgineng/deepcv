import tensorflow as tf

def prelu(inputs):
    """
    prelu activation
    """
    alphas = tf.get_variable("alphas", shape=inputs.get_shape()[-1], dtype=tf.float32, initializer=tf.constant_initializer(0.25))
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs-abs(inputs))*0.5
    return pos + neg

def cls_ohem(cls_prob, label):
    """
    计算cls loss
    :param cls_prob: 2D tensor, -shape[N,2]
    :param label: 1D tensor, -shape[N]
    :return tensor
    """
    zeros = tf.zeros_like(label)
    #将-1,-2的label变为0
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)
    # cls_prob_reshape.shape [2N,1]
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])
    label_int = tf.cast(label_filter_invalid,tf.int32)
    # num_row = N
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    row = tf.range(num_row)*2
    indices_ = row + label_int
    # 选出label为1对应的prob
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))
    # cross entropy
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    # 将part和landmark的loss剔除
    valid_inds = tf.where(label < zeros,zeros,ones)
    num_valid = tf.reduce_sum(valid_inds)
    # 取loss前70%的来计算loss
    keep_num = tf.cast(num_valid*0.7,dtype=tf.int32)
    #set 0 to invalid sample
    loss = loss * valid_inds
    loss,_ = tf.nn.top_k(loss, k=keep_num)
    return tf.reduce_mean(loss)

def bbox_ohem(bbox_pred,bbox_target,label):
    """
    计算bbox loss
    :param bbox_pred: 2D tensor -shape[N,4]
    :param bbox_target: 2D tensor -shape[N,4]
    :param label: 1D tensor -shape[N]
    :return: tensor
    """
    zeros_index = tf.zeros_like(label, dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)
    # 只有pos和part参与loss
    valid_inds = tf.where(tf.equal(tf.abs(label), 1),ones_index,zeros_index)
    # regression loss
    square_error = tf.square(bbox_pred-bbox_target)
    square_error = tf.reduce_sum(square_error,axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    #无效loss置0
    square_error = square_error*valid_inds
    # ------
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


def landmark_ohem(landmark_pred, landmark_target, label):
    # keep label =-2  then do landmark detection
    ones = tf.ones_like(label, dtype=tf.float32)
    zeros = tf.zeros_like(label, dtype=tf.float32)
    # 只计算landmark的loss
    valid_inds = tf.where(tf.equal(label, -2), ones, zeros)
    square_error = tf.square(landmark_pred - landmark_target)
    square_error = tf.reduce_sum(square_error, axis=1)
    num_valid = tf.reduce_sum(valid_inds)
    # keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    keep_num = tf.cast(num_valid, dtype=tf.int32)
    square_error = square_error * valid_inds
    _, k_index = tf.nn.top_k(square_error, k=keep_num)
    square_error = tf.gather(square_error, k_index)
    return tf.reduce_mean(square_error)


def cal_accuracy(cls_prob, label):
    pred = tf.argmax(cls_prob, axis=1)
    label_int = tf.cast(label, tf.int64)
    cond = tf.where(tf.greater_equal(label_int, 0))
    picked = tf.squeeze(cond)
    label_picked = tf.gather(label_int, picked)
    pred_picked = tf.gather(pred, picked)
    accuracy_op = tf.reduce_mean(tf.cast(tf.equal(label_picked, pred_picked), tf.float32))
    return accuracy_op