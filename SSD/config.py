IMAGE_SHAPE = (300,300)
FEAT_LAYERS=["block4", "block7", "block8", "block9", "block10", "block11"]
FEAT_SHAPES=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
ANCHOR_SIZE_BOUNDS=[0.15, 0.90]  # diff from the original paper
NUM_CLASS = 21
ANCHOR_SIZES=[(21., 45.),
              (45., 99.),
              (99., 153.),
              (153., 207.),
              (207., 261.),
              (261., 315.)]
ANCHOR_RATIOS=[[2, .5],
               [2, .5, 3, 1. / 3],
               [2, .5, 3, 1. / 3],
               [2, .5, 3, 1. / 3],
               [2, .5],
               [2, .5]]
ANCHOR_STEPS=[8, 16, 32, 64, 100, 300]
ANCHOR_OFFSET=0.5
NORMALIZATOINS=[1, -1, -1, -1, -1, -1]
PRIOR_SCALING=[0.1, 0.1, 0.2, 0.2]


VOC_LABELS = {
    'none': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow':10,
    'diningtable': 11,
    'dog':12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}