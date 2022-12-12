# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Training of segmentation model
"""
# pylint: disable = E1129 W0611
# flake8: noqa = E501
import os
import time
import argparse
import pathlib
import sys
from utils import vgg_unet, train, train_hyperparameters_tuning


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Please provide the Latest Checkpoint path e.g '
                             'for "./vgg-unet.1"...Default is None')
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        required=False,
                        default='Aerial_Semantic_Segmentation_Drone_Dataset/'
                                'dataset/semantic_drone_dataset/',
                        help='Absolute path to the dataset folder containing '
                             '"original_images" and "label_images_semantic" folders')
    parser.add_argument('-e',
                        '--epochs',
                        type=str,
                        required=False,
                        default=1,
                        help='provide the number of epochs want to train')
    parser.add_argument('-hy',
                        '--hyperparams',
                        type=str,
                        required=False,
                        default=0,
                        help='enable the hyperparameters for tuning...Default is "0"')

   
    parser.add_argument('-o',
                        '--outpath',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 to save Intel Model by creating intel folder in model path,  \
                        for stock 0')

    
    FLAGS = parser.parse_args()
    model_path = FLAGS.model_path
    train_images_path = os.path.join(FLAGS.data_path, "original_images")
    train_labels_path = os.path.join(FLAGS.data_path, "label_images_semantic")
    epochs = int(FLAGS.epochs)
    hyp_flag = int(FLAGS.hyperparams)
    path = pathlib.Path(FLAGS.model_path)
    os.makedirs(path,exist_ok=True)

    
               
    if FLAGS.model_path is None:
        print("Please provide path to save the model...\n")
        sys.exit(1)
    else:
        if FLAGS.outpath != 1:
            model_path = FLAGS.model_path + "/stock/vgg_unet"
        else:
            model_path = FLAGS.model_path + "/intel/vgg_unet"

   

    # Model Initialization

    N_CLASSES = 21  # Aerial Semantic Segmentation Drone Dataset tree,

    model = vgg_unet(n_classes=N_CLASSES, input_height=416, input_width=608)
    model_from_name = {"vgg_unet": vgg_unet}

    # Train
    if not hyp_flag:
        print("Started data validation and Training for ", epochs, " epochs")
        start_time = time.time()
        train(model=model, train_images=train_images_path,
              train_annotations=train_labels_path,
              checkpoints_path=model_path, epochs=epochs
              )
        #print("Time Taken for Training in seconds --> ", time.time()-start_time)
    else:
        print("Started Hyperprameter tuning ")
        start_time = time.time()
        total_time = train_hyperparameters_tuning(model=model, train_images=train_images_path,
                                                  train_annotations=train_labels_path, epochs=epochs,
                                                  load_weights=model_path)
        print("Time Taken for Total Hyper parameter Tuning and Model loading "
              "in seconds --> ", time.time() - start_time)
        print("total_time --> ", total_time)
