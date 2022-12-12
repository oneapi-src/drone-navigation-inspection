# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
INC QUANTIZATION model saving
"""
# pylint: disable=C0301 E0401 C0103 I1101 R0913 R1708 E0611 W0612 W0611 C0413
# flake8: noqa = E501
import os
import argparse
import itertools
import random
import six
from cv2 import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from neural_compressor.experimental import Quantization, common
from neural_compressor.experimental import Benchmark
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
tf.compat.v1.disable_eager_execution()
from utils import  vgg_unet, image_segmentation_generator, \
    get_image_array, get_segmentation_array, get_pairs_from_paths


class Dataset:
    """Creating Dataset class for getting Image and labels"""
    def __init__(self, data_root_path):
        self.data_root_path = data_root_path
        self.train_images = os.path.join(data_root_path, "original_images")
        self.train_annotations = os.path.join(data_root_path, "label_images_semantic")
        traingen = image_segmentation_generator(self.train_images, self.train_annotations, batch_size=2,
                                                n_classes=n_classes, input_height=input_height,
                                                input_width=input_width, output_height=output_height,
                                                output_width=output_width)
        self.test_images, self.test_labels = next(traingen)

    def __getitem__(self, index):
        return self.test_images[index], self.test_labels[index]

    def __len__(self):
        return len(self.test_images)

    def eval_function(self, graph_model):
        """ evaluate function to get relative accuracy of FP32"""
        n_classes = 21
        input_height = 416
        input_width = 608
        output_height = 208
        output_width = 304
        print("Model Input height , Model Input width, Model Output Height, Model Output Width")
        print(input_height, input_width, output_height, output_width)
        
        # Definfing input & output nodes of model
        INPUTS, OUTPUTS = 'input_1', 'Identity'
        output_graph = optimize_for_inference(graph_model.as_graph_def(), [INPUTS], [OUTPUTS],
                                              dtypes.float32.as_datatype_enum, False)
        
        # Initializing session
        tf.import_graph_def(output_graph, name="")
        l_input = graph_model.get_tensor_by_name('input_1:0')  # Input Tensor
        l_output = graph_model.get_tensor_by_name('Identity:0')  # Output Tensor
        config = tf.compat.v1.ConfigProto()
        sess = tf.compat.v1.Session(graph=graph_model, config=config)

        # Getting image path 
        inp_images_dir = os.path.join(self.data_root_path, "original_images")
        annotations_dir = os.path.join(self.data_root_path, "label_images_semantic")
        paths = get_pairs_from_paths(inp_images_dir, annotations_dir, mode="eval")
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

        tp = np.zeros(21)
        fp = np.zeros(21)
        fn = np.zeros(21)
        n_pixels = np.zeros(21)

        for inp, ann in tqdm(zip(inp_images, annotations)):
            x = get_image_array(inp, input_width, input_height, ordering="channels_last")
            x = np.expand_dims(x, axis=0)
            pr = sess.run(l_output, feed_dict={l_input: x})
            pr = pr[0]
            pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
            gt = get_segmentation_array(ann, n_classes, output_width, output_height, no_reshape=True)
            gt = gt.argmax(-1)
            pr = pr.flatten()
            gt = gt.flatten()

            for cl_i in range(n_classes):
                tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
                fp[cl_i] += np.sum((pr == cl_i) * (gt != cl_i))
                fn[cl_i] += np.sum((pr != cl_i) * (gt == cl_i))
                n_pixels[cl_i] += np.sum(gt == cl_i)

        cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
        n_pixels_norm = n_pixels / np.sum(n_pixels)
        frequency_weighted_iu = np.sum(cl_wise_score * n_pixels_norm)
        mean_iu = np.mean(cl_wise_score)
        return cl_wise_score[1]


# Define the command line arguments to get FP32 modelpath & Saving INT8 model path
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--modelpath',
                        type=str,
                        required=False,
                        default='./frozen_graph/frozen_graph.pb',
                        help='Model path trained with tensorflow ".pb" file')
    parser.add_argument('-o',
                        '--outpath',
                        type=str,
                        required=False,
                        default='./inc_compressed_model/output',
                        help='default output quantized model will be save in ./inc_compressed_model/output folder')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        required=False,
                        default='./deploy.yaml',
                        help='Yaml file for quantizing model, default is "./deploy.yaml"')
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
                        help='batchsize for the dataloader....default is 1')

    paramters = parser.parse_args()
    FLAGS = parser.parse_args()
    model_path = FLAGS.modelpath
    config_path = FLAGS.config
    out_path = FLAGS.outpath
    data_path = FLAGS.data_path
    batchsize = int(FLAGS.batchsize)
    
    n_classes = 21
    input_height = 416
    input_width = 608
    output_height = 208
    output_width = 304

    # Quantization 
    quantizer = Quantization(config_path)
    quantizer.model = model_path
    dataset = Dataset(data_path)
    quantizer.calib_dataloader = common.DataLoader(dataset)
    quantizer.eval_func =  dataset.eval_function
    q_model = quantizer.fit()
    q_model.save(out_path)

   
    print("******************Evaluating the FP32 Model************************")
    dataset = Dataset(data_path)
    evaluator = Benchmark(config_path)
    evaluator.model = model_path
    evaluator.b_dataloader = common.DataLoader(dataset, batch_size=batchsize)
    evaluator('performance')

    print("******************Evaluating the FP32 Model************************")
    dataset = Dataset(data_path)
    evaluator = Benchmark(config_path)
    evaluator.model = out_path
    evaluator.b_dataloader = common.DataLoader(dataset, batch_size=batchsize)
    evaluator('performance')
