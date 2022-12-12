# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
inference of FP32 Model and INT8 Model
"""
# pylint: disable=E0401 C0301 R0913 I1101 C0103 R1708 E1129
# flake8: noqa = E501
import os
import time
import argparse
import itertools
import random
from cv2 import cv2
import numpy as np
import tensorflow as tf
import six
tf.compat.v1.disable_eager_execution()


def get_image_array(image_input, width, height, imgnorm="sub_mean", ordering='channels_first'):
    """ Load image array from input """

    if isinstance(image_input, np.ndarray):
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise Exception(f"get_image_array: path {image_input} doesn't exist")
        img = cv2.imread(image_input, 1)
    else:
        raise Exception(f"get_image_array: Can't process input type {str(type(image_input))}")

    if imgnorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgnorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif imgnorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img/255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_segmentation_array(image_input, num_classes, width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, num_classes))

    if isinstance(image_input, np.ndarray):
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise Exception(f"get_segmentation_array: path {image_input} doesn't exist")
        img = cv2.imread(image_input, 1)
    else:
        raise Exception(f"get_segmentation_array: Can't process input type {str(type(image_input))}")

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(N_CLASSES):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, N_CLASSES))

    return seg_labels


def get_pairs_from_paths(images_path, segs_path, ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    acceptable_image_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    acceptable_segmentation_formats = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, dir_entry)) \
                and os.path.splitext(dir_entry)[1] in acceptable_image_formats:
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

    for dir_entry in os.listdir(segs_path):
        if os.path.isfile(os.path.join(segs_path, dir_entry)) \
                and os.path.splitext(dir_entry)[1] in acceptable_segmentation_formats:
            file_name, file_extension = os.path.splitext(dir_entry)
            if file_name in segmentation_files:
                raise Exception(f"Segmentation file with filename {file_name} already exists "
                                f"and is ambiguous to resolve with path {os.path.join(segs_path, dir_entry)}."
                                f"Please remove or rename the latter.")
            segmentation_files[file_name] = (file_extension, os.path.join(segs_path, dir_entry))

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path, segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise Exception(f"No corresponding segmentation found for image {image_full_path}.")

    return return_value


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 num_classes, in_height, in_width,
                                 out_height, out_width):
    """
    image_segmentation_generator
    """
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        x = []
        y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            x.append(get_image_array(im, in_width,
                                     in_height, ordering='channels_last'))
            y.append(get_segmentation_array(
                seg, num_classes, out_width, out_height))

        yield np.array(x), np.array(y)


class Dataset:
    """Creating Dataset class for getting Image and labels"""
    def __init__(self, data_root_path, batch_size):
        self.train_images = os.path.join(data_root_path, "original_images")
        self.train_annotations = os.path.join(data_root_path, "label_images_semantic")
        traingen = image_segmentation_generator(self.train_images, self.train_annotations, batch_size=batch_size,
                                                num_classes=N_CLASSES, in_height=INPUT_HEIGHT,
                                                in_width=INPUT_WIDTH, out_height=OUTPUT_HEIGHT,
                                                out_width=OUTPUT_WIDTH)
        self.test_images, self.test_labels = next(traingen)

    def __getitem__(self, index):
        return self.test_images, self.test_labels

    def __len__(self):
        return len(self.test_images)


# Define the command line arguments to input the Hyperparameters - batchsize & Learning Rate
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--modelpath',
                        type=str,
                        required=False,
                        default='./frozen_graph/frozen_graph.pb',
                        help='provide frozen Model path ".pb" file...users can also use INC INT8 quantized model here')
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        required=False,
                        default='Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset/',
                        help='Absolute path to the dataset folder containing '
                             '"original_images" and "label_images_semantic" folders')
    parser.add_argument('-b',
                        '--batchsize',
                        type=str,
                        required=False,
                        default=1,
                        help='batchsize used for inference')

    paramters = parser.parse_args()
    FLAGS = parser.parse_args()
    model_path = FLAGS.modelpath
    data_path = FLAGS.data_path
    batchsize = int(FLAGS.batchsize)

    N_CLASSES = 21
    INPUT_HEIGHT = 416
    INPUT_WIDTH = 608
    OUTPUT_HEIGHT = 208
    OUTPUT_WIDTH = 304

    dataset = Dataset(data_path, batch_size=batchsize)

    # Load frozen graph using TensorFlow 1.x functions
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            # Load the graph in graph_def
            print("load graph")
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(graph_def, input_map=None,
                                    return_elements=None,
                                    name="",
                                    op_dict=None,
                                    producer_op_list=None)
                l_input = graph.get_tensor_by_name('input_1:0')  # Input Tensor
                l_output = graph.get_tensor_by_name('Identity:0')  # Output Tensor
                # initialize_all_variables
                tf.compat.v1.global_variables_initializer()

                # Model warm up adjustment process for the model to reach an optimal state
                print("Model warmup initiated ")
                for i in range(5):
                    (images, labels) = next(iter(dataset))
                    Session_out = sess.run(l_output, feed_dict={l_input: images})
                print("Model warm up completed for this inference run")

                # Run Kitty model on single image
                AVG_START_TIME = 0
                for i in range(10):
                    (images, labels) = next(iter(dataset))
                    # images = np.expand_dims(images, axis=0)
                    start_time = time.time()
                    Session_out = sess.run(l_output, feed_dict={l_input: images})
                    end_time = time.time()-start_time
                    AVG_START_TIME += end_time
                    print("Time Taken for model inference in seconds ---> ", end_time)
                print("Average Time Taken for model inference in seconds ---> ", AVG_START_TIME/10)
                