from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dataset
import numpy as np
import os
import tensorflow as tf
from model import Decomposition_Net_Translation, ImageReconstruction_reflection
from warp_utils import dense_image_warp
import cv2
import glob

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_string('train_dir', 'temp_online_ckpt/',
                           """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1001,
                            """Number of steps to run.""")
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """The number of samples in each batch.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_string('training_data_path', 'imgs',
                           """Training data path.""")
tf.app.flags.DEFINE_string('training_scene', None,
                           """Training scene id in training data path.""")
tf.app.flags.DEFINE_integer('blur_size', 21,
                            """Gaussian blur kernel size used before feed images into PWC-Net""")
tf.app.flags.DEFINE_string('GPU_ID', '0',
                           """GPU ID""")

CROP_PATCH_H = 336
CROP_PATCH_W = 448
GPU_ID = FLAGS.GPU_ID

import sys

sys.path.insert(1, 'tfoptflow/tfoptflow/')
from copy import deepcopy
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS

nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = 'tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = ['/device:GPU:' + GPU_ID]
nn_opts['controller'] = '/device:GPU:' + GPU_ID
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2


def _read_image_random_size(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    # image_decoded.set_shape([256, 448, 3])
    return tf.cast(image_decoded, dtype=tf.float32) / 255.0


def _read_image_random_size_large(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    # image_decoded.set_shape([256, 448, 3])
    return tf.cast(image_decoded, dtype=tf.float32) / 255.0


def random_scaling(image, seed=1):
    scaling = tf.random_uniform([], 0.4, 0.6, seed=seed)
    return tf.image.resize_images(image, [tf.cast(tf.round(256 * scaling), tf.int32),
                                          tf.cast(tf.round(256 * scaling), tf.int32)])


def flow_to_img(flow):
    flow_magnitude = tf.sqrt(1e-6 + flow[..., 0] ** 2.0 + flow[..., 1] ** 2.0)
    flow_angle = tf.atan2(flow[..., 0], flow[..., 1])

    hsv_0 = ((flow_angle / np.pi) + 1.0) / 2.0
    hsv_1 = (flow_magnitude - tf.reduce_min(flow_magnitude, axis=[1, 2], keepdims=True)) / (
            1e-6 + tf.reduce_max(flow_magnitude, axis=[1, 2], keepdims=True) - tf.reduce_min(flow_magnitude,
                                                                                             axis=[1, 2],
                                                                                             keepdims=True))
    hsv_2 = tf.ones(tf.shape(hsv_0))
    hsv = tf.stack([hsv_0, hsv_1, hsv_2], -1)
    rgb = tf.image.hsv_to_rgb(hsv)

    return rgb


def warp(I, F, H, W):
    return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)),
                      [FLAGS.batch_size, H, W, 3])


def train():
    """resize training images into 16x"""
    if not os.path.exists('tmp'):
        os.makedirs('tmp')
    def resize_and_save(img_path):
        original_img = cv2.imread(img_path)
        NEW_H = int(np.ceil(float(original_img.shape[0]) / 16.0)) * 16
        NEW_W = int(np.ceil(float(original_img.shape[1]) / 16.0)) * 16
        new_img = cv2.resize(original_img, dsize=(NEW_W, NEW_H), interpolation=cv2.INTER_CUBIC)
        new_path = os.path.join('tmp', os.path.split(img_path)[-1])
        cv2.imwrite(new_path, new_img)
    for img_path in sorted(glob.glob(FLAGS.training_data_path + '/' + FLAGS.training_scene + '*.png')):
        resize_and_save(img_path)

    with tf.Graph().as_default():
        def get_online_data(path):
            data_list_F0 = sorted(glob.glob(path))
            dataset_F0 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_F0))
            dataset_F0 = dataset_F0.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=21, count=None, seed=6)).map(
                _read_image_random_size).map(
                lambda image: tf.random_crop(image, [CROP_PATCH_H, CROP_PATCH_W, 3], seed=6))
            dataset_F0 = dataset_F0.prefetch(16)
            return dataset_F0

        def get_online_data_large(path):
            data_list_F0 = sorted(glob.glob(path))
            dataset_F0 = tf.data.Dataset.from_tensor_slices(tf.constant(data_list_F0))
            dataset_F0 = dataset_F0.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=21, count=None, seed=6)).map(
                _read_image_random_size_large)
            dataset_F0 = dataset_F0.prefetch(16)
            return dataset_F0

        dataset_online_I0 = get_online_data('tmp/' + FLAGS.training_scene + '*I0.png')
        dataset_online_I1 = get_online_data('tmp/' + FLAGS.training_scene + '*I1.png')
        dataset_online_I2 = get_online_data('tmp/' + FLAGS.training_scene + '*I2.png')
        dataset_online_I3 = get_online_data('tmp/' + FLAGS.training_scene + '*I3.png')
        dataset_online_I4 = get_online_data('tmp/' + FLAGS.training_scene + '*I4.png')
        batch_online_I0 = dataset_online_I0.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I1 = dataset_online_I1.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I2 = dataset_online_I2.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I3 = dataset_online_I3.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I4 = dataset_online_I4.batch(FLAGS.batch_size).make_initializable_iterator()
        fused_frame0 = batch_online_I0.get_next()
        fused_frame1 = batch_online_I1.get_next()
        fused_frame2 = batch_online_I2.get_next()
        fused_frame3 = batch_online_I3.get_next()
        fused_frame4 = batch_online_I4.get_next()
        dataset_online_I0_large = get_online_data_large('tmp/' + FLAGS.training_scene + '*I0.png')
        dataset_online_I1_large = get_online_data_large('tmp/' + FLAGS.training_scene + '*I1.png')
        dataset_online_I2_large = get_online_data_large('tmp/' + FLAGS.training_scene + '*I2.png')
        dataset_online_I3_large = get_online_data_large('tmp/' + FLAGS.training_scene + '*I3.png')
        dataset_online_I4_large = get_online_data_large('tmp/' + FLAGS.training_scene + '*I4.png')
        batch_online_I0_large = dataset_online_I0_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I1_large = dataset_online_I1_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I2_large = dataset_online_I2_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I3_large = dataset_online_I3_large.batch(FLAGS.batch_size).make_initializable_iterator()
        batch_online_I4_large = dataset_online_I4_large.batch(FLAGS.batch_size).make_initializable_iterator()
        fused_frame0_large = batch_online_I0_large.get_next()
        fused_frame1_large = batch_online_I1_large.get_next()
        fused_frame2_large = batch_online_I2_large.get_next()
        fused_frame3_large = batch_online_I3_large.get_next()
        fused_frame4_large = batch_online_I4_large.get_next()

        def PWC_full(F0, F1, F2, F3, F4, B0, B1, B2, B3, B4, lvl_h, lvl_w, pwc_h, pwc_w, lvl):
            ratio_h = float(lvl_h) / float(pwc_h)
            ratio_w = float(lvl_w) / float(pwc_w)
            nn = ModelPWCNet(mode='test', options=nn_opts)
            nn.print_config()
            F0 = tf.image.resize_bilinear(F0, (pwc_h, pwc_w))
            F1 = tf.image.resize_bilinear(F1, (pwc_h, pwc_w))
            F2 = tf.image.resize_bilinear(F2, (pwc_h, pwc_w))
            F3 = tf.image.resize_bilinear(F3, (pwc_h, pwc_w))
            F4 = tf.image.resize_bilinear(F4, (pwc_h, pwc_w))
            B0 = tf.image.resize_bilinear(B0, (pwc_h, pwc_w))
            B1 = tf.image.resize_bilinear(B1, (pwc_h, pwc_w))
            B2 = tf.image.resize_bilinear(B2, (pwc_h, pwc_w))
            B3 = tf.image.resize_bilinear(B3, (pwc_h, pwc_w))
            B4 = tf.image.resize_bilinear(B4, (pwc_h, pwc_w))
            """intensity normalization"""
            F_max = tf.reduce_max(tf.concat([F0, F1, F2, F3, F4], -1), [1, 2, 3], keepdims=True)
            B_max = tf.reduce_max(tf.concat([B0, B1, B2, B3, B4], -1), [1, 2, 3], keepdims=True)
            F_min = tf.reduce_min(tf.concat([F0, F1, F2, F3, F4], -1), [1, 2, 3], keepdims=True)
            B_min = tf.reduce_min(tf.concat([B0, B1, B2, B3, B4], -1), [1, 2, 3], keepdims=True)
            F0 = (F0 - F_min) / tf.maximum((F_max - F_min), 1e-10)
            F1 = (F1 - F_min) / tf.maximum((F_max - F_min), 1e-10)
            F2 = (F2 - F_min) / tf.maximum((F_max - F_min), 1e-10)
            F3 = (F3 - F_min) / tf.maximum((F_max - F_min), 1e-10)
            F4 = (F4 - F_min) / tf.maximum((F_max - F_min), 1e-10)
            B0 = (B0 - B_min) / tf.maximum((B_max - B_min), 1e-10)
            B1 = (B1 - B_min) / tf.maximum((B_max - B_min), 1e-10)
            B2 = (B2 - B_min) / tf.maximum((B_max - B_min), 1e-10)
            B3 = (B3 - B_min) / tf.maximum((B_max - B_min), 1e-10)
            B4 = (B4 - B_min) / tf.maximum((B_max - B_min), 1e-10)
            tmp_list = []
            tmp_list.append(tf.stack([F0, F1], 1))
            tmp_list.append(tf.stack([F0, F2], 1))
            tmp_list.append(tf.stack([F0, F3], 1))
            tmp_list.append(tf.stack([F0, F4], 1))
            tmp_list.append(tf.stack([F1, F0], 1))
            tmp_list.append(tf.stack([F1, F2], 1))
            tmp_list.append(tf.stack([F1, F3], 1))
            tmp_list.append(tf.stack([F1, F4], 1))
            tmp_list.append(tf.stack([F2, F0], 1))
            tmp_list.append(tf.stack([F2, F1], 1))
            tmp_list.append(tf.stack([F2, F3], 1))
            tmp_list.append(tf.stack([F2, F4], 1))
            tmp_list.append(tf.stack([F3, F0], 1))
            tmp_list.append(tf.stack([F3, F1], 1))
            tmp_list.append(tf.stack([F3, F2], 1))
            tmp_list.append(tf.stack([F3, F4], 1))
            tmp_list.append(tf.stack([F4, F0], 1))
            tmp_list.append(tf.stack([F4, F1], 1))
            tmp_list.append(tf.stack([F4, F2], 1))
            tmp_list.append(tf.stack([F4, F3], 1))
            tmp_list.append(tf.stack([B0, B1], 1))
            tmp_list.append(tf.stack([B0, B2], 1))
            tmp_list.append(tf.stack([B0, B3], 1))
            tmp_list.append(tf.stack([B0, B4], 1))
            tmp_list.append(tf.stack([B1, B0], 1))
            tmp_list.append(tf.stack([B1, B2], 1))
            tmp_list.append(tf.stack([B1, B3], 1))
            tmp_list.append(tf.stack([B1, B4], 1))
            tmp_list.append(tf.stack([B2, B0], 1))
            tmp_list.append(tf.stack([B2, B1], 1))
            tmp_list.append(tf.stack([B2, B3], 1))
            tmp_list.append(tf.stack([B2, B4], 1))
            tmp_list.append(tf.stack([B3, B0], 1))
            tmp_list.append(tf.stack([B3, B1], 1))
            tmp_list.append(tf.stack([B3, B2], 1))
            tmp_list.append(tf.stack([B3, B4], 1))
            tmp_list.append(tf.stack([B4, B0], 1))
            tmp_list.append(tf.stack([B4, B1], 1))
            tmp_list.append(tf.stack([B4, B2], 1))
            tmp_list.append(tf.stack([B4, B3], 1))

            PWC_input = tf.concat(tmp_list, 0)  # [batch_size*20, 2, H, W, 3]
            PWC_input = tf.reshape(PWC_input, [FLAGS.batch_size * 40, 2, pwc_h, pwc_w, 3])
            pred_labels, _ = nn.nn(PWC_input, reuse=tf.AUTO_REUSE)
            print(pred_labels)

            pred_labels = tf.image.resize_bilinear(pred_labels, (lvl_h, lvl_w))
            """
            0: W
            1: H
            """
            ratio_tensor = tf.expand_dims(tf.expand_dims(
                tf.expand_dims(tf.convert_to_tensor(np.asarray([ratio_w, ratio_h]), dtype=tf.float32), 0), 0), 0)

            FF01 = pred_labels[FLAGS.batch_size * 0:FLAGS.batch_size * 1] * ratio_tensor
            FF02 = pred_labels[FLAGS.batch_size * 1:FLAGS.batch_size * 2] * ratio_tensor
            FF03 = pred_labels[FLAGS.batch_size * 2:FLAGS.batch_size * 3] * ratio_tensor
            FF04 = pred_labels[FLAGS.batch_size * 3:FLAGS.batch_size * 4] * ratio_tensor
            FF10 = pred_labels[FLAGS.batch_size * 4:FLAGS.batch_size * 5] * ratio_tensor
            FF12 = pred_labels[FLAGS.batch_size * 5:FLAGS.batch_size * 6] * ratio_tensor
            FF13 = pred_labels[FLAGS.batch_size * 6:FLAGS.batch_size * 7] * ratio_tensor
            FF14 = pred_labels[FLAGS.batch_size * 7:FLAGS.batch_size * 8] * ratio_tensor
            FF20 = pred_labels[FLAGS.batch_size * 8:FLAGS.batch_size * 9] * ratio_tensor
            FF21 = pred_labels[FLAGS.batch_size * 9:FLAGS.batch_size * 10] * ratio_tensor
            FF23 = pred_labels[FLAGS.batch_size * 10:FLAGS.batch_size * 11] * ratio_tensor
            FF24 = pred_labels[FLAGS.batch_size * 11:FLAGS.batch_size * 12] * ratio_tensor
            FF30 = pred_labels[FLAGS.batch_size * 12:FLAGS.batch_size * 13] * ratio_tensor
            FF31 = pred_labels[FLAGS.batch_size * 13:FLAGS.batch_size * 14] * ratio_tensor
            FF32 = pred_labels[FLAGS.batch_size * 14:FLAGS.batch_size * 15] * ratio_tensor
            FF34 = pred_labels[FLAGS.batch_size * 15:FLAGS.batch_size * 16] * ratio_tensor
            FF40 = pred_labels[FLAGS.batch_size * 16:FLAGS.batch_size * 17] * ratio_tensor
            FF41 = pred_labels[FLAGS.batch_size * 17:FLAGS.batch_size * 18] * ratio_tensor
            FF42 = pred_labels[FLAGS.batch_size * 18:FLAGS.batch_size * 19] * ratio_tensor
            FF43 = pred_labels[FLAGS.batch_size * 19:FLAGS.batch_size * 20] * ratio_tensor
            FB01 = pred_labels[FLAGS.batch_size * 20:FLAGS.batch_size * 21] * ratio_tensor
            FB02 = pred_labels[FLAGS.batch_size * 21:FLAGS.batch_size * 22] * ratio_tensor
            FB03 = pred_labels[FLAGS.batch_size * 22:FLAGS.batch_size * 23] * ratio_tensor
            FB04 = pred_labels[FLAGS.batch_size * 23:FLAGS.batch_size * 24] * ratio_tensor
            FB10 = pred_labels[FLAGS.batch_size * 24:FLAGS.batch_size * 25] * ratio_tensor
            FB12 = pred_labels[FLAGS.batch_size * 25:FLAGS.batch_size * 26] * ratio_tensor
            FB13 = pred_labels[FLAGS.batch_size * 26:FLAGS.batch_size * 27] * ratio_tensor
            FB14 = pred_labels[FLAGS.batch_size * 27:FLAGS.batch_size * 28] * ratio_tensor
            FB20 = pred_labels[FLAGS.batch_size * 28:FLAGS.batch_size * 29] * ratio_tensor
            FB21 = pred_labels[FLAGS.batch_size * 29:FLAGS.batch_size * 30] * ratio_tensor
            FB23 = pred_labels[FLAGS.batch_size * 30:FLAGS.batch_size * 31] * ratio_tensor
            FB24 = pred_labels[FLAGS.batch_size * 31:FLAGS.batch_size * 32] * ratio_tensor
            FB30 = pred_labels[FLAGS.batch_size * 32:FLAGS.batch_size * 33] * ratio_tensor
            FB31 = pred_labels[FLAGS.batch_size * 33:FLAGS.batch_size * 34] * ratio_tensor
            FB32 = pred_labels[FLAGS.batch_size * 34:FLAGS.batch_size * 35] * ratio_tensor
            FB34 = pred_labels[FLAGS.batch_size * 35:FLAGS.batch_size * 36] * ratio_tensor
            FB40 = pred_labels[FLAGS.batch_size * 36:FLAGS.batch_size * 37] * ratio_tensor
            FB41 = pred_labels[FLAGS.batch_size * 37:FLAGS.batch_size * 38] * ratio_tensor
            FB42 = pred_labels[FLAGS.batch_size * 38:FLAGS.batch_size * 39] * ratio_tensor
            FB43 = pred_labels[FLAGS.batch_size * 39:FLAGS.batch_size * 40] * ratio_tensor

            FF01 = tf.stop_gradient(FF01)
            FF02 = tf.stop_gradient(FF02)
            FF03 = tf.stop_gradient(FF03)
            FF04 = tf.stop_gradient(FF04)
            FF10 = tf.stop_gradient(FF10)
            FF12 = tf.stop_gradient(FF12)
            FF13 = tf.stop_gradient(FF13)
            FF14 = tf.stop_gradient(FF14)
            FF20 = tf.stop_gradient(FF20)
            FF21 = tf.stop_gradient(FF21)
            FF23 = tf.stop_gradient(FF23)
            FF24 = tf.stop_gradient(FF24)
            FF30 = tf.stop_gradient(FF30)
            FF31 = tf.stop_gradient(FF31)
            FF32 = tf.stop_gradient(FF32)
            FF34 = tf.stop_gradient(FF34)
            FF40 = tf.stop_gradient(FF40)
            FF41 = tf.stop_gradient(FF41)
            FF42 = tf.stop_gradient(FF42)
            FF43 = tf.stop_gradient(FF43)
            FB01 = tf.stop_gradient(FB01)
            FB02 = tf.stop_gradient(FB02)
            FB03 = tf.stop_gradient(FB03)
            FB04 = tf.stop_gradient(FB04)
            FB10 = tf.stop_gradient(FB10)
            FB12 = tf.stop_gradient(FB12)
            FB13 = tf.stop_gradient(FB13)
            FB14 = tf.stop_gradient(FB14)
            FB20 = tf.stop_gradient(FB20)
            FB21 = tf.stop_gradient(FB21)
            FB23 = tf.stop_gradient(FB23)
            FB24 = tf.stop_gradient(FB24)
            FB30 = tf.stop_gradient(FB30)
            FB31 = tf.stop_gradient(FB31)
            FB32 = tf.stop_gradient(FB32)
            FB34 = tf.stop_gradient(FB34)
            FB40 = tf.stop_gradient(FB40)
            FB41 = tf.stop_gradient(FB41)
            FB42 = tf.stop_gradient(FB42)
            FB43 = tf.stop_gradient(FB43)

            return FF01, FF02, FF03, FF04, \
                   FF10, FF12, FF13, FF14, \
                   FF20, FF21, FF23, FF24, \
                   FF30, FF31, FF32, FF34, \
                   FF40, FF41, FF42, FF43, \
                   FB01, FB02, FB03, FB04, \
                   FB10, FB12, FB13, FB14, \
                   FB20, FB21, FB23, FB24, \
                   FB30, FB31, FB32, FB34, \
                   FB40, FB41, FB42, FB43

        model = Decomposition_Net_Translation(CROP_PATCH_H // 16, CROP_PATCH_W // 16, False, False, False)
        FF01_4, FF02_4, FF03_4, FF04_4, \
        FF10_4, FF12_4, FF13_4, FF14_4, \
        FF20_4, FF21_4, FF23_4, FF24_4, \
        FF30_4, FF31_4, FF32_4, FF34_4, \
        FF40_4, FF41_4, FF42_4, FF43_4, \
        FB01_4, FB02_4, FB03_4, FB04_4, \
        FB10_4, FB12_4, FB13_4, FB14_4, \
        FB20_4, FB21_4, FB23_4, FB24_4, \
        FB30_4, FB31_4, FB32_4, FB34_4, \
        FB40_4, FB41_4, FB42_4, FB43_4 = model.inference(fused_frame0_large, fused_frame1_large, fused_frame2_large,
                                                         fused_frame3_large, fused_frame4_large)

        flows = []
        flows.append((FF01_4, FF02_4, FF03_4, FF04_4, \
                      FF10_4, FF12_4, FF13_4, FF14_4, \
                      FF20_4, FF21_4, FF23_4, FF24_4, \
                      FF30_4, FF31_4, FF32_4, FF34_4, \
                      FF40_4, FF41_4, FF42_4, FF43_4, \
                      FB01_4, FB02_4, FB03_4, FB04_4, \
                      FB10_4, FB12_4, FB13_4, FB14_4, \
                      FB20_4, FB21_4, FB23_4, FB24_4, \
                      FB30_4, FB31_4, FB32_4, FB34_4, \
                      FB40_4, FB41_4, FB42_4, FB43_4))

        """image"""
        model4 = ImageReconstruction_reflection(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=4)
        F0_pred_4, F1_pred_4, F2_pred_4, F3_pred_4, F4_pred_4, \
        B0_pred_4, B1_pred_4, B2_pred_4, B3_pred_4, B4_pred_4 = model4._build_model(tf.concat([fused_frame0,
                                                                                               fused_frame1,
                                                                                               fused_frame2,
                                                                                               fused_frame3,
                                                                                               fused_frame4], 3),
                                                                                    None, None, None, None, None,
                                                                                    None, None, None, None, None,
                                                                                    FF01_4, FF02_4, FF03_4, FF04_4,
                                                                                    FF10_4, FF12_4, FF13_4, FF14_4,
                                                                                    FF20_4, FF21_4, FF23_4, FF24_4,
                                                                                    FF30_4, FF31_4, FF32_4, FF34_4,
                                                                                    FF40_4, FF41_4, FF42_4, FF43_4,
                                                                                    FB01_4, FB02_4, FB03_4, FB04_4,
                                                                                    FB10_4, FB12_4, FB13_4, FB14_4,
                                                                                    FB20_4, FB21_4, FB23_4, FB24_4,
                                                                                    FB30_4, FB31_4, FB32_4, FB34_4,
                                                                                    FB40_4, FB41_4, FB42_4, FB43_4)

        """upsample (no resize in model)"""
        F0_pred_4_up = tf.image.resize_bilinear(F0_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        F1_pred_4_up = tf.image.resize_bilinear(F1_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        F2_pred_4_up = tf.image.resize_bilinear(F2_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        F3_pred_4_up = tf.image.resize_bilinear(F3_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        F4_pred_4_up = tf.image.resize_bilinear(F4_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        B0_pred_4_up = tf.image.resize_bilinear(B0_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        B1_pred_4_up = tf.image.resize_bilinear(B1_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        B2_pred_4_up = tf.image.resize_bilinear(B2_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        B3_pred_4_up = tf.image.resize_bilinear(B3_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        B4_pred_4_up = tf.image.resize_bilinear(B4_pred_4, (CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3)))
        FF01_3, FF02_3, FF03_3, FF04_3, \
        FF10_3, FF12_3, FF13_3, FF14_3, \
        FF20_3, FF21_3, FF23_3, FF24_3, \
        FF30_3, FF31_3, FF32_3, FF34_3, \
        FF40_3, FF41_3, FF42_3, FF43_3, \
        FB01_3, FB02_3, FB03_3, FB04_3, \
        FB10_3, FB12_3, FB13_3, FB14_3, \
        FB20_3, FB21_3, FB23_3, FB24_3, \
        FB30_3, FB31_3, FB32_3, FB34_3, \
        FB40_3, FB41_3, FB42_3, FB43_3 = PWC_full(F0_pred_4_up, F1_pred_4_up, F2_pred_4_up, F3_pred_4_up, F4_pred_4_up,
                                                  B0_pred_4_up, B1_pred_4_up, B2_pred_4_up, B3_pred_4_up, B4_pred_4_up,
                                                  CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3),
                                                  int(np.ceil(float(CROP_PATCH_H // (2 ** 3)) / 64.0)) * 64,
                                                  int(np.ceil(float(CROP_PATCH_W // (2 ** 3)) / 64.0)) * 64, 3)
        flows.append((FF01_3, FF02_3, FF03_3, FF04_3, \
                      FF10_3, FF12_3, FF13_3, FF14_3, \
                      FF20_3, FF21_3, FF23_3, FF24_3, \
                      FF30_3, FF31_3, FF32_3, FF34_3, \
                      FF40_3, FF41_3, FF42_3, FF43_3, \
                      FB01_3, FB02_3, FB03_3, FB04_3, \
                      FB10_3, FB12_3, FB13_3, FB14_3, \
                      FB20_3, FB21_3, FB23_3, FB24_3, \
                      FB30_3, FB31_3, FB32_3, FB34_3, \
                      FB40_3, FB41_3, FB42_3, FB43_3))

        """3"""
        model3 = ImageReconstruction_reflection(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=3)
        F0_pred_3, F1_pred_3, F2_pred_3, F3_pred_3, F4_pred_3, \
        B0_pred_3, B1_pred_3, B2_pred_3, B3_pred_3, B4_pred_3 = model3._build_model(tf.concat([fused_frame0,
                                                                                               fused_frame1,
                                                                                               fused_frame2,
                                                                                               fused_frame3,
                                                                                               fused_frame4], 3),
                                                                                    F0_pred_4_up, F1_pred_4_up,
                                                                                    F2_pred_4_up,
                                                                                    F3_pred_4_up, F4_pred_4_up,
                                                                                    B0_pred_4_up, B1_pred_4_up,
                                                                                    B2_pred_4_up,
                                                                                    B3_pred_4_up, B4_pred_4_up,
                                                                                    FF01_3, FF02_3, FF03_3, FF04_3,
                                                                                    FF10_3, FF12_3, FF13_3, FF14_3,
                                                                                    FF20_3, FF21_3, FF23_3, FF24_3,
                                                                                    FF30_3, FF31_3, FF32_3, FF34_3,
                                                                                    FF40_3, FF41_3, FF42_3, FF43_3,
                                                                                    FB01_3, FB02_3, FB03_3, FB04_3,
                                                                                    FB10_3, FB12_3, FB13_3, FB14_3,
                                                                                    FB20_3, FB21_3, FB23_3, FB24_3,
                                                                                    FB30_3, FB31_3, FB32_3, FB34_3,
                                                                                    FB40_3, FB41_3, FB42_3, FB43_3)

        F0_pred_3_up = tf.image.resize_bilinear(F0_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        F1_pred_3_up = tf.image.resize_bilinear(F1_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        F2_pred_3_up = tf.image.resize_bilinear(F2_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        F3_pred_3_up = tf.image.resize_bilinear(F3_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        F4_pred_3_up = tf.image.resize_bilinear(F4_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        B0_pred_3_up = tf.image.resize_bilinear(B0_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        B1_pred_3_up = tf.image.resize_bilinear(B1_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        B2_pred_3_up = tf.image.resize_bilinear(B2_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        B3_pred_3_up = tf.image.resize_bilinear(B3_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        B4_pred_3_up = tf.image.resize_bilinear(B4_pred_3, (CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2)))
        FF01_2, FF02_2, FF03_2, FF04_2, \
        FF10_2, FF12_2, FF13_2, FF14_2, \
        FF20_2, FF21_2, FF23_2, FF24_2, \
        FF30_2, FF31_2, FF32_2, FF34_2, \
        FF40_2, FF41_2, FF42_2, FF43_2, \
        FB01_2, FB02_2, FB03_2, FB04_2, \
        FB10_2, FB12_2, FB13_2, FB14_2, \
        FB20_2, FB21_2, FB23_2, FB24_2, \
        FB30_2, FB31_2, FB32_2, FB34_2, \
        FB40_2, FB41_2, FB42_2, FB43_2 = PWC_full(F0_pred_3_up, F1_pred_3_up, F2_pred_3_up, F3_pred_3_up, F4_pred_3_up,
                                                  B0_pred_3_up, B1_pred_3_up, B2_pred_3_up, B3_pred_3_up, B4_pred_3_up,
                                                  CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2),
                                                  int(np.ceil(float(CROP_PATCH_H // (2 ** 2)) / 64.0)) * 64,
                                                  int(np.ceil(float(CROP_PATCH_W // (2 ** 2)) / 64.0)) * 64, 2)
        flows.append((FF01_2, FF02_2, FF03_2, FF04_2, \
                      FF10_2, FF12_2, FF13_2, FF14_2, \
                      FF20_2, FF21_2, FF23_2, FF24_2, \
                      FF30_2, FF31_2, FF32_2, FF34_2, \
                      FF40_2, FF41_2, FF42_2, FF43_2, \
                      FB01_2, FB02_2, FB03_2, FB04_2, \
                      FB10_2, FB12_2, FB13_2, FB14_2, \
                      FB20_2, FB21_2, FB23_2, FB24_2, \
                      FB30_2, FB31_2, FB32_2, FB34_2, \
                      FB40_2, FB41_2, FB42_2, FB43_2))
        """2"""
        model2 = ImageReconstruction_reflection(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=2)
        F0_pred_2, F1_pred_2, F2_pred_2, F3_pred_2, F4_pred_2, \
        B0_pred_2, B1_pred_2, B2_pred_2, B3_pred_2, B4_pred_2 = model2._build_model(tf.concat([fused_frame0,
                                                                                               fused_frame1,
                                                                                               fused_frame2,
                                                                                               fused_frame3,
                                                                                               fused_frame4], 3),
                                                                                    F0_pred_3_up, F1_pred_3_up,
                                                                                    F2_pred_3_up,
                                                                                    F3_pred_3_up, F4_pred_3_up,
                                                                                    B0_pred_3_up, B1_pred_3_up,
                                                                                    B2_pred_3_up,
                                                                                    B3_pred_3_up, B4_pred_3_up,
                                                                                    FF01_2, FF02_2, FF03_2, FF04_2,
                                                                                    FF10_2, FF12_2, FF13_2, FF14_2,
                                                                                    FF20_2, FF21_2, FF23_2, FF24_2,
                                                                                    FF30_2, FF31_2, FF32_2, FF34_2,
                                                                                    FF40_2, FF41_2, FF42_2, FF43_2,
                                                                                    FB01_2, FB02_2, FB03_2, FB04_2,
                                                                                    FB10_2, FB12_2, FB13_2, FB14_2,
                                                                                    FB20_2, FB21_2, FB23_2, FB24_2,
                                                                                    FB30_2, FB31_2, FB32_2, FB34_2,
                                                                                    FB40_2, FB41_2, FB42_2, FB43_2)

        F0_pred_2_up = tf.image.resize_bilinear(F0_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        F1_pred_2_up = tf.image.resize_bilinear(F1_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        F2_pred_2_up = tf.image.resize_bilinear(F2_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        F3_pred_2_up = tf.image.resize_bilinear(F3_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        F4_pred_2_up = tf.image.resize_bilinear(F4_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        B0_pred_2_up = tf.image.resize_bilinear(B0_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        B1_pred_2_up = tf.image.resize_bilinear(B1_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        B2_pred_2_up = tf.image.resize_bilinear(B2_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        B3_pred_2_up = tf.image.resize_bilinear(B3_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        B4_pred_2_up = tf.image.resize_bilinear(B4_pred_2, (CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1)))
        FF01_1, FF02_1, FF03_1, FF04_1, \
        FF10_1, FF12_1, FF13_1, FF14_1, \
        FF20_1, FF21_1, FF23_1, FF24_1, \
        FF30_1, FF31_1, FF32_1, FF34_1, \
        FF40_1, FF41_1, FF42_1, FF43_1, \
        FB01_1, FB02_1, FB03_1, FB04_1, \
        FB10_1, FB12_1, FB13_1, FB14_1, \
        FB20_1, FB21_1, FB23_1, FB24_1, \
        FB30_1, FB31_1, FB32_1, FB34_1, \
        FB40_1, FB41_1, FB42_1, FB43_1 = PWC_full(F0_pred_2_up, F1_pred_2_up, F2_pred_2_up, F3_pred_2_up, F4_pred_2_up,
                                                  B0_pred_2_up, B1_pred_2_up, B2_pred_2_up, B3_pred_2_up, B4_pred_2_up,
                                                  CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1),
                                                  int(np.ceil(float(CROP_PATCH_H // (2 ** 1)) / 64.0)) * 64,
                                                  int(np.ceil(float(CROP_PATCH_W // (2 ** 1)) / 64.0)) * 64, 1)
        flows.append((FF01_1, FF02_1, FF03_1, FF04_1, \
                      FF10_1, FF12_1, FF13_1, FF14_1, \
                      FF20_1, FF21_1, FF23_1, FF24_1, \
                      FF30_1, FF31_1, FF32_1, FF34_1, \
                      FF40_1, FF41_1, FF42_1, FF43_1, \
                      FB01_1, FB02_1, FB03_1, FB04_1, \
                      FB10_1, FB12_1, FB13_1, FB14_1, \
                      FB20_1, FB21_1, FB23_1, FB24_1, \
                      FB30_1, FB31_1, FB32_1, FB34_1, \
                      FB40_1, FB41_1, FB42_1, FB43_1))

        """1"""
        model1 = ImageReconstruction_reflection(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=1)
        F0_pred_1, F1_pred_1, F2_pred_1, F3_pred_1, F4_pred_1, \
        B0_pred_1, B1_pred_1, B2_pred_1, B3_pred_1, B4_pred_1 = model1._build_model(tf.concat([fused_frame0,
                                                                                               fused_frame1,
                                                                                               fused_frame2,
                                                                                               fused_frame3,
                                                                                               fused_frame4], 3),
                                                                                    F0_pred_2_up, F1_pred_2_up,
                                                                                    F2_pred_2_up,
                                                                                    F3_pred_2_up, F4_pred_2_up,
                                                                                    B0_pred_2_up, B1_pred_2_up,
                                                                                    B2_pred_2_up,
                                                                                    B3_pred_2_up, B4_pred_2_up,
                                                                                    FF01_1, FF02_1, FF03_1, FF04_1,
                                                                                    FF10_1, FF12_1, FF13_1, FF14_1,
                                                                                    FF20_1, FF21_1, FF23_1, FF24_1,
                                                                                    FF30_1, FF31_1, FF32_1, FF34_1,
                                                                                    FF40_1, FF41_1, FF42_1, FF43_1,
                                                                                    FB01_1, FB02_1, FB03_1, FB04_1,
                                                                                    FB10_1, FB12_1, FB13_1, FB14_1,
                                                                                    FB20_1, FB21_1, FB23_1, FB24_1,
                                                                                    FB30_1, FB31_1, FB32_1, FB34_1,
                                                                                    FB40_1, FB41_1, FB42_1, FB43_1)

        F0_pred_1_up = tf.image.resize_bilinear(F0_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        F1_pred_1_up = tf.image.resize_bilinear(F1_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        F2_pred_1_up = tf.image.resize_bilinear(F2_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        F3_pred_1_up = tf.image.resize_bilinear(F3_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        F4_pred_1_up = tf.image.resize_bilinear(F4_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        B0_pred_1_up = tf.image.resize_bilinear(B0_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        B1_pred_1_up = tf.image.resize_bilinear(B1_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        B2_pred_1_up = tf.image.resize_bilinear(B2_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        B3_pred_1_up = tf.image.resize_bilinear(B3_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        B4_pred_1_up = tf.image.resize_bilinear(B4_pred_1, (CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0)))
        FF01_0, FF02_0, FF03_0, FF04_0, \
        FF10_0, FF12_0, FF13_0, FF14_0, \
        FF20_0, FF21_0, FF23_0, FF24_0, \
        FF30_0, FF31_0, FF32_0, FF34_0, \
        FF40_0, FF41_0, FF42_0, FF43_0, \
        FB01_0, FB02_0, FB03_0, FB04_0, \
        FB10_0, FB12_0, FB13_0, FB14_0, \
        FB20_0, FB21_0, FB23_0, FB24_0, \
        FB30_0, FB31_0, FB32_0, FB34_0, \
        FB40_0, FB41_0, FB42_0, FB43_0 = PWC_full(F0_pred_1_up, F1_pred_1_up, F2_pred_1_up, F3_pred_1_up, F4_pred_1_up,
                                                  B0_pred_1_up, B1_pred_1_up, B2_pred_1_up, B3_pred_1_up, B4_pred_1_up,
                                                  CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0),
                                                  int(np.ceil(float(CROP_PATCH_H // (2 ** 0)) / 64.0)) * 64,
                                                  int(np.ceil(float(CROP_PATCH_W // (2 ** 0)) / 64.0)) * 64, 0)
        flows.append((FF01_0, FF02_0, FF03_0, FF04_0, \
                      FF10_0, FF12_0, FF13_0, FF14_0, \
                      FF20_0, FF21_0, FF23_0, FF24_0, \
                      FF30_0, FF31_0, FF32_0, FF34_0, \
                      FF40_0, FF41_0, FF42_0, FF43_0, \
                      FB01_0, FB02_0, FB03_0, FB04_0, \
                      FB10_0, FB12_0, FB13_0, FB14_0, \
                      FB20_0, FB21_0, FB23_0, FB24_0, \
                      FB30_0, FB31_0, FB32_0, FB34_0, \
                      FB40_0, FB41_0, FB42_0, FB43_0))

        """0"""
        model0 = ImageReconstruction_reflection(FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, level=0)
        F0_pred_0, F1_pred_0, F2_pred_0, F3_pred_0, F4_pred_0, \
        B0_pred_0, B1_pred_0, B2_pred_0, B3_pred_0, B4_pred_0 = model0._build_model(tf.concat([fused_frame0,
                                                                                               fused_frame1,
                                                                                               fused_frame2,
                                                                                               fused_frame3,
                                                                                               fused_frame4], 3),
                                                                                    F0_pred_1_up, F1_pred_1_up,
                                                                                    F2_pred_1_up,
                                                                                    F3_pred_1_up, F4_pred_1_up,
                                                                                    B0_pred_1_up, B1_pred_1_up,
                                                                                    B2_pred_1_up,
                                                                                    B3_pred_1_up, B4_pred_1_up,
                                                                                    FF01_0, FF02_0, FF03_0, FF04_0,
                                                                                    FF10_0, FF12_0, FF13_0, FF14_0,
                                                                                    FF20_0, FF21_0, FF23_0, FF24_0,
                                                                                    FF30_0, FF31_0, FF32_0, FF34_0,
                                                                                    FF40_0, FF41_0, FF42_0, FF43_0,
                                                                                    FB01_0, FB02_0, FB03_0, FB04_0,
                                                                                    FB10_0, FB12_0, FB13_0, FB14_0,
                                                                                    FB20_0, FB21_0, FB23_0, FB24_0,
                                                                                    FB30_0, FB31_0, FB32_0, FB34_0,
                                                                                    FB40_0, FB41_0, FB42_0, FB43_0)

        F_pred = []
        F_pred.append(tf.concat([F0_pred_0, F1_pred_0, F2_pred_0, F3_pred_0, F4_pred_0], -1))
        F_pred.append(tf.concat([F0_pred_1, F1_pred_1, F2_pred_1, F3_pred_1, F4_pred_1], -1))
        F_pred.append(tf.concat([F0_pred_2, F1_pred_2, F2_pred_2, F3_pred_2, F4_pred_2], -1))
        F_pred.append(tf.concat([F0_pred_3, F1_pred_3, F2_pred_3, F3_pred_3, F4_pred_3], -1))
        F_pred.append(tf.concat([F0_pred_4, F1_pred_4, F2_pred_4, F3_pred_4, F4_pred_4], -1))
        B_pred = []
        B_pred.append(tf.concat([B0_pred_0, B1_pred_0, B2_pred_0, B3_pred_0, B4_pred_0], -1))
        B_pred.append(tf.concat([B0_pred_1, B1_pred_1, B2_pred_1, B3_pred_1, B4_pred_1], -1))
        B_pred.append(tf.concat([B0_pred_2, B1_pred_2, B2_pred_2, B3_pred_2, B4_pred_2], -1))
        B_pred.append(tf.concat([B0_pred_3, B1_pred_3, B2_pred_3, B3_pred_3, B4_pred_3], -1))
        B_pred.append(tf.concat([B0_pred_4, B1_pred_4, B2_pred_4, B3_pred_4, B4_pred_4], -1))

        """full size PWC"""

        def generate_gaussian_kernel(sz):
            kernel = cv2.getGaussianKernel(sz, 0)
            kernel = np.dot(kernel, kernel.transpose())
            return tf.cast(kernel[:, :, np.newaxis, np.newaxis], tf.float32)

        if FLAGS.blur_size >= 1:
            kernel = generate_gaussian_kernel(FLAGS.blur_size)

        def apply_gaussian_blur_image(x):
            x = tf.pad(x, [[0, 0], [40, 40], [40, 40], [0, 0]], 'SYMMETRIC')
            x_0 = tf.nn.conv2d(x[..., 0:1], kernel, strides=[1, 1, 1, 1], padding="SAME")
            x_1 = tf.nn.conv2d(x[..., 1:2], kernel, strides=[1, 1, 1, 1], padding="SAME")
            x_2 = tf.nn.conv2d(x[..., 2:3], kernel, strides=[1, 1, 1, 1], padding="SAME")
            output = tf.concat([x_0, x_1, x_2], -1)
            return output[:, 40:-40, 40:-40]

        if FLAGS.blur_size >= 1:
            F0_pred_0_blur = apply_gaussian_blur_image(F0_pred_0)
            F1_pred_0_blur = apply_gaussian_blur_image(F1_pred_0)
            F2_pred_0_blur = apply_gaussian_blur_image(F2_pred_0)
            F3_pred_0_blur = apply_gaussian_blur_image(F3_pred_0)
            F4_pred_0_blur = apply_gaussian_blur_image(F4_pred_0)
            B0_pred_0_blur = apply_gaussian_blur_image(B0_pred_0)
            B1_pred_0_blur = apply_gaussian_blur_image(B1_pred_0)
            B2_pred_0_blur = apply_gaussian_blur_image(B2_pred_0)
            B3_pred_0_blur = apply_gaussian_blur_image(B3_pred_0)
            B4_pred_0_blur = apply_gaussian_blur_image(B4_pred_0)
        else:
            F0_pred_0_blur = F0_pred_0
            F1_pred_0_blur = F1_pred_0
            F2_pred_0_blur = F2_pred_0
            F3_pred_0_blur = F3_pred_0
            F4_pred_0_blur = F4_pred_0
            B0_pred_0_blur = B0_pred_0
            B1_pred_0_blur = B1_pred_0
            B2_pred_0_blur = B2_pred_0
            B3_pred_0_blur = B3_pred_0
            B4_pred_0_blur = B4_pred_0
        FF01_, FF02_, FF03_, FF04_, \
        FF10_, FF12_, FF13_, FF14_, \
        FF20_, FF21_, FF23_, FF24_, \
        FF30_, FF31_, FF32_, FF34_, \
        FF40_, FF41_, FF42_, FF43_, \
        FB01_, FB02_, FB03_, FB04_, \
        FB10_, FB12_, FB13_, FB14_, \
        FB20_, FB21_, FB23_, FB24_, \
        FB30_, FB31_, FB32_, FB34_, \
        FB40_, FB41_, FB42_, FB43_ = PWC_full(F0_pred_0_blur, F1_pred_0_blur, F2_pred_0_blur, F3_pred_0_blur,
                                              F4_pred_0_blur,
                                              B0_pred_0_blur, B1_pred_0_blur, B2_pred_0_blur, B3_pred_0_blur,
                                              B4_pred_0_blur,
                                              CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0),
                                              int(np.ceil(float(CROP_PATCH_H // (2 ** 0)) / 64.0)) * 64,
                                              int(np.ceil(float(CROP_PATCH_W // (2 ** 0)) / 64.0)) * 64, 3)

        loss = 0
        loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0]
        for i in range(len(F_pred)):
            # for i in range(2):
            # i = 0
            _, h, w, _ = tf.unstack(tf.shape(F_pred[i]))
            print('level: ' + str(i))
            print(h)
            print(w)
            I0_lvl = tf.image.resize_bilinear(fused_frame0, [h, w])
            I1_lvl = tf.image.resize_bilinear(fused_frame1, [h, w])
            I2_lvl = tf.image.resize_bilinear(fused_frame2, [h, w])
            I3_lvl = tf.image.resize_bilinear(fused_frame3, [h, w])
            I4_lvl = tf.image.resize_bilinear(fused_frame4, [h, w])

            def compute_loss_2(FF02, FB02, FF12, FB12, FF32, FB32, FF42, FB42, F2, B2, I2_lvl, I0_lvl, I1_lvl, I3_lvl,
                               I4_lvl):
                sub_loss = 0

                """convert largest flow to lvl"""
                FF02 = tf.image.resize_bilinear(FF02 / (2.0 ** i), [h, w])
                FB02 = tf.image.resize_bilinear(FB02 / (2.0 ** i), [h, w])
                FF12 = tf.image.resize_bilinear(FF12 / (2.0 ** i), [h, w])
                FB12 = tf.image.resize_bilinear(FB12 / (2.0 ** i), [h, w])
                FF32 = tf.image.resize_bilinear(FF32 / (2.0 ** i), [h, w])
                FB32 = tf.image.resize_bilinear(FB32 / (2.0 ** i), [h, w])
                FF42 = tf.image.resize_bilinear(FF42 / (2.0 ** i), [h, w])
                FB42 = tf.image.resize_bilinear(FB42 / (2.0 ** i), [h, w])

                """warping consistency loss"""
                sub_loss += (loss_weight[i] * tf.reduce_mean(
                    tf.abs(I0_lvl - warp(F2, FF02, h, w) - warp(B2, FB02, h, w))))
                sub_loss += (loss_weight[i] * tf.reduce_mean(
                    tf.abs(I1_lvl - warp(F2, FF12, h, w) - warp(B2, FB12, h, w))))
                sub_loss += (loss_weight[i] * tf.reduce_mean(tf.abs(I2_lvl - F2 - B2)))
                sub_loss += (loss_weight[i] * tf.reduce_mean(
                    tf.abs(I3_lvl - warp(F2, FF32, h, w) - warp(B2, FB32, h, w))))
                sub_loss += (loss_weight[i] * tf.reduce_mean(
                    tf.abs(I4_lvl - warp(F2, FF42, h, w) - warp(B2, FB42, h, w))))

                """TV loss"""
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(F2[:, 1:] - F2[:, :-1]))))
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(F2[:, :, 1:] - F2[:, :, :-1]))))
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(B2[:, 1:] - B2[:, :-1]))))
                sub_loss += (loss_weight[i] * (0.1 * tf.reduce_mean(tf.abs(B2[:, :, 1:] - B2[:, :, :-1]))))
                return sub_loss

            """full size PWC"""
            loss += compute_loss_2(FF02_, FB02_, FF12_, FB12_, FF32_, FB32_, FF42_, FB42_,
                                   tf.clip_by_value(F_pred[i][..., 6:9], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 6:9], 0.0, 1.0),
                                   I2_lvl, I0_lvl, I1_lvl, I3_lvl, I4_lvl)
            loss += compute_loss_2(FF10_, FB10_, FF20_, FB20_, FF30_, FB30_, FF40_, FB40_,
                                   tf.clip_by_value(F_pred[i][..., 0:3], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 0:3], 0.0, 1.0),
                                   I0_lvl, I1_lvl, I2_lvl, I3_lvl, I4_lvl)
            loss += compute_loss_2(FF01_, FB01_, FF21_, FB21_, FF31_, FB31_, FF41_, FB41_,
                                   tf.clip_by_value(F_pred[i][..., 3:6], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 3:6], 0.0, 1.0),
                                   I1_lvl, I0_lvl, I2_lvl, I3_lvl, I4_lvl)
            loss += compute_loss_2(FF03_, FB03_, FF13_, FB13_, FF23_, FB23_, FF43_, FB43_,
                                   tf.clip_by_value(F_pred[i][..., 9:12], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 9:12], 0.0, 1.0),
                                   I3_lvl, I0_lvl, I1_lvl, I2_lvl, I4_lvl)
            loss += compute_loss_2(FF04_, FB04_, FF14_, FB14_, FF24_, FB24_, FF34_, FB34_,
                                   tf.clip_by_value(F_pred[i][..., 12:15], 0.0, 1.0),
                                   tf.clip_by_value(B_pred[i][..., 12:15], 0.0, 1.0),
                                   I4_lvl, I0_lvl, I1_lvl, I2_lvl, I3_lvl)

        t_vars = tf.all_variables()
        print('all layers:')
        for var in t_vars: print(var.name)
        dof_vars = [var for var in t_vars if 'FusionLayer_' in var.name]
        # dof_vars = [var for var in t_vars if 'FusionLayer_F_0' in var.name or 'FusionLayer_B_0' in var.name or 'FusionLayer_F_1' in var.name or 'FusionLayer_B_1' in var.name]
        print('optimize layers:')
        for var in dof_vars: print(var.name)

        # Perform learning rate scheduling.
        learning_rate = FLAGS.initial_learning_rate

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #     update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=dof_vars)

        # Create an optimizer that performs gradient descent.
        with tf.variable_scope(tf.get_variable_scope(), reuse=None):
            update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=dof_vars)

        tf.summary.scalar('loss', loss)
        tf.summary.image('fused_frame0', fused_frame0, 3)
        tf.summary.image('fused_frame1', fused_frame1, 3)
        tf.summary.image('fused_frame2', fused_frame2, 3)
        tf.summary.image('fused_frame3', fused_frame3, 3)
        tf.summary.image('fused_frame4', fused_frame4, 3)
        tf.summary.image('fused_frame0_large', fused_frame0_large, 3)
        tf.summary.image('fused_frame1_large', fused_frame1_large, 3)
        tf.summary.image('fused_frame2_large', fused_frame2_large, 3)
        tf.summary.image('fused_frame3_large', fused_frame3_large, 3)
        tf.summary.image('fused_frame4_large', fused_frame4_large, 3)

        tf.summary.image('B2_pred_4', B2_pred_4, 3)
        tf.summary.image('F2_pred_4', F2_pred_4, 3)
        tf.summary.image('B2_pred_3', B2_pred_3, 3)
        tf.summary.image('F2_pred_3', F2_pred_3, 3)
        tf.summary.image('B2_pred_2', B2_pred_2, 3)
        tf.summary.image('F2_pred_2', F2_pred_2, 3)
        tf.summary.image('B2_pred_1', B2_pred_1, 3)
        tf.summary.image('F2_pred_1', F2_pred_1, 3)
        tf.summary.image('B2_pred_0', B2_pred_0, 3)
        tf.summary.image('F2_pred_0', F2_pred_0, 3)

        # Create a saver.
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=50)

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge_all()

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run([init,
                  batch_online_I0.initializer, batch_online_I1.initializer, batch_online_I2.initializer,
                  batch_online_I3.initializer, batch_online_I4.initializer,
                  batch_online_I0_large.initializer, batch_online_I1_large.initializer,
                  batch_online_I2_large.initializer, batch_online_I3_large.initializer,
                  batch_online_I4_large.initializer])

        saver2 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "pwcnet" in v.name])
        saver2.restore(sess, nn_opts['ckpt_path'])
        saver4 = tf.train.Saver(var_list=[v for v in tf.all_variables() if
                                          "FeaturePyramidExtractor" in v.name or "TranslationEstimator" in v.name])
        saver4.restore(sess, 'ckpt_decomposition_reflection/model.ckpt')
        saver5 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "FusionLayer_" in v.name])
        saver5.restore(sess, 'ckpt_reconstruction_reflection/model.ckpt')

        # Summary Writter
        summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir,
            graph=sess.graph)

        for step in range(0, FLAGS.max_steps):

            # Run single step update.
            _, loss_value = sess.run([update_op, loss])

            if step % 10 == 0:
                print("Loss at step %d: %f" % (step, loss_value))

            if step % 100 == 0:
                # Output Summary
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save checkpoint
            if step % 1000 == 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    train()
