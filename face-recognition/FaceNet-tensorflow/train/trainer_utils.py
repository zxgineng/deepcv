import numpy as np


def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.split('#', 1)[0]
            if line:
                par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    return learning_rate


def sample_people(dataset, sample_size):
    """
    随机sample图片直到足量
    :param dataset:  list of ImageClass, ImageClass,tuple,包含一种类别和其下的所有图片路径列表
    :return: image_paths: list of str, 所有path
              num_per_class: list of int, path中每个class的数量
    """
    # Sample classes from the dataset
    nrof_classes = len(dataset)
    class_indices = np.arange(nrof_classes)
    np.random.shuffle(class_indices)

    i = 0
    # 记录所有path
    image_paths = []
    num_per_class = []
    # 记录sample对应的class_index
    sampled_class_indices = []
    # Sample images from these classes until we have enough
    while len(image_paths) < sample_size:
        class_index = class_indices[i]
        nrof_images_in_class = len(dataset[class_index])
        image_indices = np.arange(nrof_images_in_class)
        # shuffle类别下的图片路径
        np.random.shuffle(image_indices)
        nrof_images_from_class = min(nrof_images_in_class,sample_size - len(image_paths))
        idx = image_indices[0:nrof_images_from_class]
        image_paths_for_class = [dataset[class_index].image_paths[j] for j in idx]
        sampled_class_indices += [class_index] * nrof_images_from_class
        image_paths += image_paths_for_class
        num_per_class.append(nrof_images_from_class)
        i += 1

    return image_paths, num_per_class


def select_triplets(embeddings, nrof_images_per_class, image_paths,alpha):
    """
    遍历所有anchor和pos，在满足的neg内随机选择组成triplet
    :param embeddings: 2D tensor, shape:[2000,128]
    :param nrof_images_per_class: list of int,image_paths中对应的class按序统计
    :param image_paths: list of str，2000图片路径
    :param alpha: margin
    :return:
    """
    trip_idx = 0
    emb_start_idx = 0
    num_trips = 0
    triplets = []

    for nrof_images in nrof_images_per_class:
        # 为一类中的所有图片寻找triplet
        for j in range(1, nrof_images):
            a_idx = emb_start_idx + j - 1
            # 所有距离 shape:[2000]
            neg_dists_sqr = np.sum(np.square(embeddings[a_idx] - embeddings), 1)
            # 遍历剩下所有的同类图片作为pos
            for pair in range(j, nrof_images):  # For every possible positive pair.
                p_idx = emb_start_idx + pair
                # 与单个pos的距离
                pos_dist_sqr = np.sum(np.square(embeddings[a_idx] - embeddings[p_idx]))
                # 所有距离中非neg的变为nan
                neg_dists_sqr[emb_start_idx:emb_start_idx + nrof_images] = np.NaN
                # 所有neg中满足与单个pos距离的H_index
                all_neg = np.where(neg_dists_sqr - pos_dist_sqr < alpha)[0]
                nrof_random_negs = all_neg.shape[0]
                # 在满足的neg中随机挑选一个 与当前的anchor,pos组成triplet
                if nrof_random_negs > 0:
                    rnd_idx = np.random.randint(nrof_random_negs)
                    n_idx = all_neg[rnd_idx]
                    triplets.append((image_paths[a_idx], image_paths[p_idx], image_paths[n_idx]))
                    trip_idx += 1

                num_trips += 1

        emb_start_idx += nrof_images

    np.random.shuffle(triplets)
    return triplets, num_trips, len(triplets)