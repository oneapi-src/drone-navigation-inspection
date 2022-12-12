# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""
Support file for training and evaluation
"""
# pylint: disable=E0401 C0301 C0103 E1101 E0611 R0913 R0914 R1708 R0912 R0915 E1129 W0612
# flake8: noqa = E501
import os
from random import SystemRandom
import itertools
import time
import glob
import json
from types import MethodType
from tqdm import tqdm
import six
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Permute, Conv2D, \
    BatchNormalization, ZeroPadding2D, concatenate, UpSampling2D, MaxPooling2D, Reshape

IMAGE_ORDERING_CHANNELS_FIRST = "channels_first"
IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
# Default IMAGE_ORDERING = channels_last
IMAGE_ORDERING = IMAGE_ORDERING_CHANNELS_LAST

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1

if IMAGE_ORDERING == 'channels_first':
    pretrained_url = "https://github.com/fchollet/" \
                     "deep-learning-models/releases/download/v0.1/" \
                     "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
elif IMAGE_ORDERING == 'channels_last':
    pretrained_url = "https://github.com/fchollet/" \
                     "deep-learning-models/releases/download/v0.1/" \
                     "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

cryptogen = SystemRandom()
class_colors = [(cryptogen.randint(0, 255), cryptogen.randint(
    0, 255), cryptogen.randint(0, 255)) for _ in range(5000)]


def get_colored_segmentation_image(seg_arr, n_classes, colors=None):
    """ get_colored_segmentation_image """
    if colors is None:
        colors = class_colors
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_img[:, :, 0] += ((seg_arr[:, :] == c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr[:, :] == c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr[:, :] == c) * (colors[c][2])).astype('uint8')

    return seg_img


def visualize_segmentation(seg_arr, inp_img=None, n_classes=None, colors=None,
                           overlay_img=False, prediction_width=None, prediction_height=None):
    """ visualize_segmentation """
    if colors is None:
        colors = class_colors
    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if prediction_height is not None and prediction_width is not None:
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img, (prediction_width, prediction_height))

    if overlay_img:
        if inp_img is not None:
            # seg_img = overlay_seg_image(inp_img, seg_img)
            pass

    return seg_img


def get_image_array(image_input, width, height, img_norm="sub_mean",
                    ordering='channels_first'):
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

    if img_norm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif img_norm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
        img = img[:, :, ::-1]
    elif img_norm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def get_segmentation_array(image_input, nclasses, width, height, no_reshape=False):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nclasses))

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

    for c in range(nclasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width * height, nclasses))

    return seg_labels


def image_segmentation_generator(images_path, segs_path, batch_size,
                                 n_classes, input_height, input_width,
                                 output_height, output_width,
                                 do_augment=False):
    """ image_segmentation_generator """
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    cryptogen.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        x = []
        y = []
        for _ in range(batch_size):
            im, seg = next(zipped)

            im = cv2.imread(im, 1)
            seg = cv2.imread(seg, 1)

            if do_augment:
                # im, seg[:, :, 0] = augment_seg(im, seg[:, :, 0], augmentation_name=augmentation_name)
                pass

            x.append(get_image_array(im, input_width,
                                     input_height, ordering=IMAGE_ORDERING))
            y.append(get_segmentation_array(
                seg, n_classes, output_width, output_height))

        yield np.array(x), np.array(y)


def get_pairs_from_paths(images_path, segs_path, mode="train", ignore_non_matching=False):
    """ Find all the images from the images_path directory and
        the segmentation images from the segs_path directory
        while checking integrity of data """

    acceptable_img_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    acceptable_segmentation_formats = [".png", ".bmp"]

    image_files = []
    segmentation_files = {}
    if mode == "train":
        for dir_entry in os.listdir(images_path)[:int(len(os.listdir(images_path)) * 0.80)]:
            if os.path.isfile(os.path.join(images_path, dir_entry)) and os.path.splitext(dir_entry)[1] \
                    in acceptable_img_formats:
                file_name, file_extension = os.path.splitext(dir_entry)
                image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

                file_extension = acceptable_segmentation_formats[0]
                if file_name in segmentation_files:
                    raise Exception(
                        f"Segmentation file with filename {file_name} "
                        f"already exists and is ambiguous "
                        f"to resolve with path {os.path.join(segs_path, dir_entry)}"
                        f". Please remove or rename the latter.")
                segmentation_files[file_name] = (file_extension,
                                                 os.path.join(segs_path, dir_entry.split(".")[0] +
                                                              acceptable_segmentation_formats[0]))
        print("80% of Data is considered for Training ===> ", int(len(os.listdir(segs_path)) * 0.80))
    else:
        for dir_entry in os.listdir(images_path)[-int(len(os.listdir(images_path)) * 0.2):]:
            if os.path.isfile(os.path.join(images_path, dir_entry)) and os.path.splitext(dir_entry)[1] \
                    in acceptable_img_formats:
                file_name, file_extension = os.path.splitext(dir_entry)
                image_files.append((file_name, file_extension, os.path.join(images_path, dir_entry)))

                file_extension = acceptable_segmentation_formats[0]
                if file_name in segmentation_files:
                    raise Exception(
                        f"Segmentation file with filename {file_name} "
                        f"already exists and is ambiguous "
                        f"to resolve with path {os.path.join(segs_path, dir_entry)}"
                        f". Please remove or rename the latter.")
                segmentation_files[file_name] = (file_extension,
                                                 os.path.join(segs_path, dir_entry.split(".")[0] +
                                                              acceptable_segmentation_formats[0]))
        print("20% of Data is considered for Evaluating===> ", int(len(os.listdir(segs_path)) * 0.2))
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


def verify_segmentation_dataset(images_path, segs_path, n_classes, show_all_errors=False):
    """ verify_segmentation_dataset """
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
        if len(img_seg_pairs) <= 0:
            print(f"Couldn't load any data from images_path: {images_path, segs_path} and segmentations path: {1}")
            return False

        return_value = True
        for im_fn, seg_fn in tqdm(img_seg_pairs):
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print(
                    f"The size of image {im_fn} and its segmentation {seg_fn} doesn't "
                    f"match (possibly the files are corrupt).")
                if not show_all_errors:
                    break
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print(
                        f"The pixel values of the segmentation image {seg_fn} violating "
                        f"range [0, {str(n_classes - 1)}]. Found maximum pixel value {max_pixel_value}")
                    if not show_all_errors:
                        break
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except RuntimeError:
        print("Found error during data loading")
        return False


def evaluate(model=None, inp_images=None, annotations=None, inp_images_dir=None, annotations_dir=None,
             checkpoints_path=None):
    """ evaluate """
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width
    print("Model Input height , Model Input width, Model Output Height, Model Output Width")
    print(input_height, input_width, output_height, output_width)

    if checkpoints_path is not None:
        with open(checkpoints_path + "_config.json", "w", encoding="utf8") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if checkpoints_path is not None:
        latest_checkpoint = checkpoints_path  # find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if inp_images is None:
        paths = get_pairs_from_paths(inp_images_dir, annotations_dir, mode="eval")
        paths = list(zip(*paths))
        inp_images = list(paths[0])
        annotations = list(paths[1])

    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)

    for inp, ann in tqdm(zip(inp_images, annotations)):
        pr = predict(model, inp)
        gt = get_segmentation_array(ann, model.n_classes, model.output_width, model.output_height, no_reshape=True)
        gt = gt.argmax(-1)
        pr = pr.flatten()
        gt = gt.flatten()

        for cl_i in range(model.n_classes):
            tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
            fp[cl_i] += np.sum((pr == cl_i) * (gt != cl_i))
            fn[cl_i] += np.sum((pr != cl_i) * (gt == cl_i))
            n_pixels[cl_i] += np.sum(gt == cl_i)

    cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
    n_pixels_norm = n_pixels / np.sum(n_pixels)
    frequency_weighted_iu = np.sum(cl_wise_score * n_pixels_norm)
    mean_iu = np.mean(cl_wise_score)
    return {"frequency_weighted_IU": frequency_weighted_iu, "mean_IU": mean_iu, "class_wise_IU": cl_wise_score}


def frozen(model=None, inp_images=None, annotations=None, inp_images_dir=None, annotations_dir=None,checkpoints_path=None):
    """ evaluate """
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width
    print("Model Input height , Model Input width, Model Output Height, Model Output Width")
    print(input_height, input_width, output_height, output_width)

    # Load frozen graph using TensorFlow 1.x functions
    with tf.Graph().as_default() as graph:
        with tf.compat.v1.Session() as sess:
            # Load the graph in graph_def
            print("load graph")
            with tf.io.gfile.GFile(checkpoints_path, "rb") as f:
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

                if inp_images is None:
                    paths = get_pairs_from_paths(inp_images_dir, annotations_dir, mode="eval")
                    paths = list(zip(*paths))
                    inp_images = list(paths[0])
                    annotations = list(paths[1])

                tp = np.zeros(model.n_classes)
                fp = np.zeros(model.n_classes)
                fn = np.zeros(model.n_classes)
                n_pixels = np.zeros(model.n_classes)

                for inp, ann in tqdm(zip(inp_images, annotations)):
                    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
                    x = np.expand_dims(x, axis=0)
                    pr = sess.run(l_output, feed_dict={l_input: x})
                    pr = pr[0]
                    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
                    gt = get_segmentation_array(ann, model.n_classes, model.output_width, model.output_height, no_reshape=True)
                    gt = gt.argmax(-1)
                    pr = pr.flatten()
                    gt = gt.flatten()

                    for cl_i in range(model.n_classes):
                        tp[cl_i] += np.sum((pr == cl_i) * (gt == cl_i))
                        fp[cl_i] += np.sum((pr == cl_i) * (gt != cl_i))
                        fn[cl_i] += np.sum((pr != cl_i) * (gt == cl_i))
                        n_pixels[cl_i] += np.sum(gt == cl_i)

                cl_wise_score = tp / (tp + fp + fn + 0.000000000001)
                n_pixels_norm = n_pixels / np.sum(n_pixels)
                frequency_weighted_iu = np.sum(cl_wise_score * n_pixels_norm)
                mean_iu = np.mean(cl_wise_score)
    return {"frequency_weighted_IU": frequency_weighted_iu, "mean_IU": mean_iu, "class_wise_IU": cl_wise_score}

    
def predict_multiple(model=None, inps=None, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     colors=None, prediction_width=None,
                     prediction_height=None):
    """ predict_multiple """
    if colors is None:
        colors = class_colors
    if model is None and (checkpoints_path is not None):
        # model = model_from_checkpoint_path(checkpoints_path)
        pass

    if inps is None and (inp_dir is not None):
        inps = glob.glob(os.path.join(inp_dir, "*.jpg")) + glob.glob(
            os.path.join(inp_dir, "*.png")) + \
               glob.glob(os.path.join(inp_dir, "*.jpeg"))

    all_prs = []

    for i, inp in enumerate(tqdm(inps)):
        if out_dir is None:
            out_fname = None
        else:
            if isinstance(inp, six.string_types):
                out_fname = os.path.join(out_dir, os.path.basename(inp))
            else:
                out_fname = os.path.join(out_dir, str(i) + ".jpg")

        pr = predict(model, inp, out_fname,
                     overlay_img=overlay_img,
                     colors=colors, prediction_width=prediction_width,
                     prediction_height=prediction_height)

        all_prs.append(pr)

    return all_prs


def predict(model=None, inp=None, out_fname=None, checkpoints_path=None, overlay_img=False,
            colors=None, prediction_width=None, prediction_height=None):
    """ predict """
    if colors is None:
        colors = class_colors
    if model is None and (checkpoints_path is not None):
        # model = model_from_checkpoint_path(checkpoints_path)
        pass

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp)

    output_width = model.output_width
    output_height = model.output_height
    input_width = model.input_width
    input_height = model.input_height
    n_classes = model.n_classes

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = visualize_segmentation(pr, inp, n_classes=n_classes, colors=colors,
                                     overlay_img=overlay_img, prediction_width=prediction_width,
                                     prediction_height=prediction_height)

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def train_hyperparameters_tuning(model, train_images, train_annotations, batch_size=4,
                                 steps_per_epoch=128, do_augment=False, epochs=3, load_weights=None):
    """
    Hyperparameter Tuning
    """

    # hyper-parameterss considered for tuning DL arch
    options = {
        "lr": [0.001, 0.01, 0.0001],
        "optimizer": ["Adam", "adadelta", "rmsprop"],
        "loss": ["categorical_crossentropy"]}

    # Replicating GridsearchCV functionality for params generation
    keys = options.keys()
    values = (options[key] for key in keys)
    p_combinations = []
    for combination in itertools.product(*values):
        if len(combination) > 0:
            p_combinations.append(combination)

    steps_per_epoch = int(steps_per_epoch / batch_size)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width
    print("Model Input height , Model Input width, Model Output Height, Model Output Width")
    print(input_height, input_width, output_height, output_width)
    print("Batch Size used for Training --> ", batch_size)

    train_gen = image_segmentation_generator(
        train_images, train_annotations, batch_size, n_classes,
        input_height, input_width, output_height, output_width, do_augment=do_augment)

    print("Total number of fits = ", len(p_combinations))
    print("Take Break!!!\nThis will take time!")
    ctr = 0
    total_time = 0
    best_config = {"accuracy":0, "best_fit":0}
    for combination in p_combinations:
        if load_weights is not None and len(load_weights) > 0:
            print("Loading weights from ", load_weights)
            model.load_weights(load_weights)

        if len(combination) > 0:
            ctr += 1
            print("Current fit is at ", ctr)
            learning_r, optimizer, loss = combination
            print("Current fit parameters --> epochs=", epochs, " learning rate=", learning_r,
                  " optimizer=", optimizer, " loss=", loss)
            if optimizer == "Adam":
                optimizer = keras.optimizers.Adam(learning_rate=learning_r)
            elif optimizer == "adadelta":
                optimizer = keras.optimizers.Adadelta(learning_rate=learning_r)
            else:
                optimizer = keras.optimizers.RMSprop(learning_rate=learning_r)

            model.compile(loss=loss,
                          optimizer=optimizer,
                          metrics=['accuracy', 'mae', keras.metrics.MeanIoU(num_classes=21)])

            start_time = time.time()
            hist=model.fit_generator(train_gen, steps_per_epoch, epochs=epochs, workers=1, use_multiprocessing=False)
            #model.fit_generator(train_gen, steps_per_epoch, epochs=epochs, workers=1, use_multiprocessing=False)
            total_time += time.time()-start_time
            print("Fit number: ", ctr, " ==> Time Taken for Training in seconds --> ", time.time()-start_time)
            if best_config["accuracy"] < hist.history["accuracy"][0]:          
                best_config["accuracy"]=hist.history["accuracy"][0]
                best_config["best_fit"]=combination
            print("The best Tuningparameter combination is :",best_config)
    return total_time


def train(model, train_images, train_annotations, verify_dataset=False,
          checkpoints_path=None, epochs=5, batch_size=4, validate=False,
          val_images=None, val_annotations=None, val_batch_size=4,
          auto_resume_checkpoint=False, load_weights=None, steps_per_epoch=512, val_steps_per_epoch=512,
          gen_use_multiprocessing=False, ignore_zero_class=False, optimizer_name='Adam', do_augment=False
          ):
    """ train """
    steps_per_epoch = int(steps_per_epoch / batch_size)

    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width
    print("Model Input height , Model Input width, Model Output Height, Model Output Width")
    print(input_height, input_width, output_height, output_width)
    print("Batch Size used for Training --> ", batch_size)
    print("Batch Size used for Validation --> ", val_batch_size)

    if optimizer_name is not None:

        if ignore_zero_class:
            loss_k = 'masked_categorical_crossentropy'
        else:
            loss_k = 'categorical_crossentropy'
        if optimizer_name == "Adam":
            optimizer_name = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss=loss_k,
                      optimizer=optimizer_name,
                      metrics=['accuracy'])

    if checkpoints_path is not None:
        with open("train_config.json", "w", encoding="utf8") as f:
            json.dump({
                "model_class": model.model_name,
                "n_classes": n_classes,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width
            }, f)

    if load_weights is not None and len(load_weights) > 0:
        print("Loading weights from ", load_weights)
        model.load_weights(load_weights)

    if auto_resume_checkpoint and (checkpoints_path is not None):
        latest_checkpoint = checkpoints_path  # find_latest_checkpoint(checkpoints_path)
        if latest_checkpoint is not None:
            print("Loading the weights from latest checkpoint ",
                  latest_checkpoint)
            model.load_weights(latest_checkpoint)

    if verify_dataset:
        print("Verifying training dataset")
        verify_segmentation_dataset(train_images, train_annotations, n_classes)
        if validate:
            print("Verifying validation dataset")
            verify_segmentation_dataset(val_images, val_annotations, n_classes)

    train_gen = image_segmentation_generator(
        train_images, train_annotations, batch_size, n_classes,
        input_height, input_width, output_height, output_width, do_augment=do_augment)
    val_gen = None
    if validate:
        val_gen = image_segmentation_generator(
            val_images, val_annotations, val_batch_size,
            n_classes, input_height, input_width, output_height, output_width)
    if checkpoints_path is None:
        checkpoints_path = "vgg-unet"
    start_time = time.time()
    if not validate:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch, epochs=1, use_multiprocessing=False)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path)
                print("saved ", checkpoints_path)
            print("Finished Epoch", ep)
    else:
        for ep in range(epochs):
            print("Starting Epoch ", ep)
            model.fit_generator(train_gen, steps_per_epoch,
                                validation_data=val_gen,
                                validation_steps=val_steps_per_epoch, epochs=1,
                                use_multiprocessing=gen_use_multiprocessing)
            if checkpoints_path is not None:
                model.save_weights(checkpoints_path)
                print("saved ", checkpoints_path)
            print("Finished Epoch", ep)
    print("Time Taken for Training in seconds --> ", time.time() - start_time)


def get_segmentation_model(input_data, output):
    """ get_segmentation_model """
    output_width, output_height, n_classes, input_height, input_width = None, None, None, None, None
    img_input = input_data
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    if IMAGE_ORDERING == 'channels_first':
        output_height = o_shape[2]
        output_width = o_shape[3]
        input_height = i_shape[2]
        input_width = i_shape[3]
        n_classes = o_shape[1]
        o = (Reshape((-1, output_height * output_width)))(o)
        o = (Permute((2, 1)))(o)
    elif IMAGE_ORDERING == 'channels_last':
        output_height = o_shape[1]
        output_width = o_shape[2]
        input_height = i_shape[1]
        input_width = i_shape[2]
        n_classes = o_shape[3]
        o = (Reshape((output_height * output_width, -1)))(o)

    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict, model)
    model.predict_multiple = MethodType(predict_multiple, model)
    model.evaluate_segmentation = MethodType(evaluate, model)

    return model


def get_vgg_encoder(input_height=224, input_width=224, pretrained='imagenet'):
    """ get_vgg_encoder """
    img_input = None
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
                     data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',
                     data_format=IMAGE_ORDERING)(x)
    f5 = x

    if pretrained == 'imagenet':
        vgg_weights_path = keras.utils.get_file(pretrained_url.rsplit('/', maxsplit=1)[-1], pretrained_url)
        Model(img_input, x).load_weights(vgg_weights_path)

    return img_input, [f1, f2, f3, f4, f5]


def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416,
          input_width=608):
    """ _unet """
    img_input, levels = encoder(
        input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4, _] = levels

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same', data_format=IMAGE_ORDERING)(o)

    model = get_segmentation_model(img_input, o)

    return model


def vgg_unet(n_classes, input_height=416, input_width=608):
    """ vgg_unet """
    model = _unet(n_classes, get_vgg_encoder, input_height=input_height, input_width=input_width)
    model.model_name = "vgg_unet"
    return model
