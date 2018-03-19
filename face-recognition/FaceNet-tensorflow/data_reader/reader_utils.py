import os
import numpy as np

def get_dataset(path):
    """
    获取所有类别与对应的图片路径
    :param path: str, data-dir
    :return: dataset: list of ImageClass, ImageClass包含一种类别和其下的所有图片路径列表
    """
    dataset = []
    path_exp = os.path.expanduser(path)
    classes = [path for path in os.listdir(path_exp) \
               if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    nrof_classes = len(classes)
    # 获取每个类别文件夹下的图片路径
    for i in range(nrof_classes):
        class_name = classes[i]
        facedir = os.path.join(path_exp, class_name)
        # 获取图片路径
        # -- list of str
        image_paths = get_image_paths(facedir)
        dataset.append(ImageClass(class_name, image_paths))

    return dataset

def get_image_paths(facedir):
    """
    读取类别文件夹下的图片，返回图片路径列表
    :param facedir: str, 类别文件夹路径
    :return: image_paths: list of dir, 图片路径列表
    """
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret