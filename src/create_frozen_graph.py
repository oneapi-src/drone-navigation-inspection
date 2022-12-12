# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

"""
Convert checkpoint to frozen graph
"""
# pylint: disable=E0401 C0301 E0611 E1136
# flake8: noqa = E501
import argparse
import sys
import os
import pathlib
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utils import vgg_unet


if __name__ == "__main__":
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Please provide the Latest Checkpoint path e.g for "./vgg-unet.1"...Default is None')


    parser.add_argument('-o',
                        '--output_saved_dir',
                        default=None,
                        type=str,
                        required=True,
                        help="directory to save frozen graph to."
                        )
    FLAGS = parser.parse_args()
    model_path = FLAGS.model_path
    output_saved_dir = FLAGS.output_saved_dir
    N_CLASSES = 21
    model = vgg_unet(n_classes=N_CLASSES, input_height=416, input_width=608)

    # Loading weights of Trained Model
    if model_path is not None:
        latest_checkpoint = model_path  # find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)
    else:
        print("Please Check the checkpoint model path Provided")
        sys.exit()
    # Convert Keras model to ConcreteFunction
    full_model = tf.function(model)
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input_1"))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    path = pathlib.Path(FLAGS.output_saved_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,logdir='.',
        	      name=os.path.join(FLAGS.output_saved_dir, "frozen_graph.pb"),
                      as_text=False
    		      )
                      