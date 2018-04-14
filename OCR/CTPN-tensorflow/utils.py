import yaml
import json

class ConfigMeta(type):
    class __ConfigMeta:

        def __init__(self, is_new=False):
            self.is_new = is_new
            self.config = None
            self.description = None
            if is_new is False:
                self.config = self.parse_yaml(self.read_fname)

        def __call__(self, fname):
            self.is_new = False
            self.config = self.parse_yaml(fname)
            self.read_fname = fname

        def parse_yaml(self, path):
            config = self.parse_description_then_remove(path)
            return yaml.load(config)

        def parse_description_then_remove(self, path):
            self.description = {}
            config = ""
            with open(path, 'r') as infile:
                for line in infile.readlines():
                    config += line
            return config

        def to_dict(self):
            return self.config

        def get(self, name, default=None):
            try:
                return self.__getattr__(name)
            except KeyError as ke:
                return default

        def __getattr__(self, name):
            self._set_config()

            config_value = self.config[name]
            if type(config_value) == dict:
                return SubConfig(config_value, get_tag=name)
            else:
                return config_value

        def __repr__(self):
            if self.config is None:
                raise FileNotFoundError("No such files start filename")
            else:
                return f"Read config file name: {self.read_fname}\n" + json.dumps(self.config, indent=4)

        def _set_config(self):
            if self.config is None:
                self.is_new = False
                self.config = self.read_file(self.base_fname)

    instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            cls.instance = cls.__ConfigMeta(is_new=True)
        return cls.instance


class Config(metaclass=ConfigMeta):
    pass


class SubConfig:
    def __init__(self, *args, get_tag=None):
        self.get_tag = get_tag
        self.__dict__ = dict(*args)

    def __getattr__(self, name):
        if name in self.__dict__["__dict__"]:
            item = self.__dict__["__dict__"][name]
            return item
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self.__dict__[name] = value
        if name != "get" and name != "__dict__":
            origin_config = Config.config
            gets = self.get_tag.split(".")
            for get in gets:
                origin_config = origin_config[get]

            origin_config[name] = value

    def get(self, name, default=None):
        return self.__dict__["__dict__"].get(name, default)

    def to_dict(self):
        return self.__dict__["__dict__"]

    def __repr__(self):
        return json.dumps(self.__dict__["__dict__"], indent=4)
#
# import math
#
# def create_anchors(shape):
#     num_anchors = len(Config.model.anchor_height)
#     num_height = math.floor(shape[0] / Config.model.anchor_width)
#     num_width = math.floor(shape[1] / Config.model.anchor_width)
#     gridy, gridx = np.mgrid[0:num_height, 0:num_width]
#     cy = (gridy + 0.5) * Config.model.anchor_width
#     cx = (gridx + 0.5) * Config.model.anchor_width
#     cy = np.expand_dims(cy, -1)
#     cx = np.expand_dims(cx, -1)
#     w = np.ones(num_anchors, dtype='float32') * Config.model.anchor_width
#     h = np.array(Config.model.anchor_height, dtype='float32')
#     return cy, cx, h, w
#
# def decode_coords(coords,anchors):
#     """decode"""
#     cy, cx, h, w = anchors
#     new_coords = np.zeros([coords.shape[0],coords.shape[1],coords.shape[2],4])
#     cy = coords[:,:,:,0] * h + cy
#     h = np.exp(coords[:,:,:,1]) * h
#     # [ymin,xmin,ymax,xmax]
#     new_coords[:,:,:,0] = cy - h/2
#     new_coords[:,:,:,1] = cx - w/2
#     new_coords[:,:,:,2] = cy + h/2
#     new_coords[:,:,:,3] = cx + w/2
#     return new_coords
#
#
# import cv2
import numpy as np
# import matplotlib.pyplot as plt
#
def hook_formatter(values):
    pred = values['pred']
    labels = values['labels']
    # a = np.array((pred==labels)).sum()
    return pred.sum()
#     image = values['image']
#     image = image + np.array([0.5])
#     image = image * 255
#     image = image.astype(np.uint8)
#     image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
#     coords = values['coords']
#     fpmask = values['fpmask']
#     fnmask = values['fnmask']
#     nmask = values['nmask']
#     pmask = values['pmask']
#     anchors = create_anchors(image.shape)
#     cy, cx, h, w = anchors
#     new_coords = np.zeros([coords.shape[0], coords.shape[1], coords.shape[2], 4])
#     new_coords[:,:,:,0] = cy - h/2
#     new_coords[:,:,:,1] = cx - w/2
#     new_coords[:,:,:,2] = cy + h/2
#     new_coords[:,:,:,3] = cx + w/2
#
#     coords = new_coords.reshape(-1,4).astype(int)
#     pcoords = coords[pmask]
#     ncoords = coords[nmask]
#     for coord in pcoords:
#         cv2.rectangle(image,(coord[1],coord[0]),(coord[3],coord[2]),(0,255,0),1)
#     for coord in ncoords:
#         cv2.rectangle(image,(coord[1],coord[0]),(coord[3],coord[2]),(0,0,255),1)
#     cv2.imshow('image',image)
#     cv2.waitKey()
#     cv2.destroyAllWindows()
#
#     return 'next'


