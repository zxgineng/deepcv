"""Performs face alignment and stores face thumbnails in the output directory."""
import argparse
import sys
from data_reader import align_dataset_mtcnn

def main(args):
    align_dataset_mtcnn.start_align(args)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, help='Directory with unaligned images.', default='E:/dataset/lfw')
    parser.add_argument('--output_dir', type=str, help='Directory with aligned face thumbnails.',
                        default='E:/dataset/aligned')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=32)
    parser.add_argument('--random_order',
                        help='Shuffles the order of images to enable alignment using multiple processes.',
                        action='store_true')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--detect_multiple_faces', type=bool,
                        help='Detect and align multiple faces per image.', default=False)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

