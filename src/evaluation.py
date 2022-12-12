# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Evaluating of segmentation model
"""
# pylint: disable = C0301 W0611 C0413
# flake8: noqa = E501
import os
import argparse
import time
import itertools
import random
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from cv2 import cv2
import numpy as np
import six
from utils import predict, evaluate, vgg_unet, frozen


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Please provide the Latest Checkpoint path e.g for "./vgg-unet.1"...Default is None')
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        required=False,
                        default='Aerial_Semantic_Segmentation_Drone_Dataset/dataset/semantic_drone_dataset/',
                        help='Absolute path to the dataset folder containing '
                             '"original_images" and "label_images_semantic" folders')
    parser.add_argument('-t',
                        '--model_type',
                        type=int,
                        required=False,
                        default='0',
                        help='0 for checkpoint '
                             '1 for frozen_graph ')
    FLAGS = parser.parse_args()
    model_path = FLAGS.model_path
    model_type=FLAGS.model_type
    data_path=FLAGS.data_path
    N_CLASSES = 21
    INPUT_HEIGHT = 416
    INPUT_WIDTH = 608
    OUTPUT_HEIGHT = 208
    OUTPUT_WIDTH = 304


    train_images_path = os.path.join(FLAGS.data_path, "original_images")
    train_labels_path = os.path.join(FLAGS.data_path, "label_images_semantic")

    # Model Initialization
    N_CLASSES = 21  # Aerial Semantic Segmentation Drone Dataset tree,

    model = vgg_unet(n_classes=N_CLASSES, input_height=416, input_width=608)
    model_from_name = {"vgg_unet": vgg_unet}
    if model_type == 0 :

        out = evaluate(model=model, inp_images_dir=train_images_path, annotations_dir=train_labels_path,
                       checkpoints_path=model_path)
        start = time.time()
        n_classes_names = ["unlabeled", "[TARGET CLASS] paved-area", "dirt", "grass", "gravel", "water", "rocks",
                       "pool", "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole",
                       "person", "dog", "car", "bicycle", "tree", "bald-tree"]
        for i, elem in enumerate(out['class_wise_IU']):
            print(n_classes_names[i], f"=>  {(elem * 100):.2f}")

        input_image = train_images_path+"/002.jpg"
        out = predict(model=model, inp=input_image, out_fname="out.png")
        print("Time Taken for Prediction in seconds --> ", time.time()-start)
    else:
        print("Inferening using pb grah",model_path)
        out = frozen(model=model, inp_images_dir=train_images_path, annotations_dir=train_labels_path,
                       checkpoints_path=model_path)
        start = time.time()
        n_classes_names = ["unlabeled", "[TARGET CLASS] paved-area", "dirt", "grass", "gravel", "water", "rocks",
                       "pool", "vegetation", "roof", "wall", "window", "door", "fence", "fence-pole",
                       "person", "dog", "car", "bicycle", "tree", "bald-tree"]
        for i, elem in enumerate(out['class_wise_IU']):
            print(n_classes_names[i], f"=>  {(elem * 100):.2f}")
            