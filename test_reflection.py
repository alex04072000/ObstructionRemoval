from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from warp_utils import dense_image_warp
import cv2
from model import Decomposition_Net_Translation, ImageReconstruction_reflection

FLAGS = tf.app.flags.FLAGS

# Define necessary FLAGS
tf.app.flags.DEFINE_integer('batch_size', 1,
                            """The number of samples in each batch.""")
tf.app.flags.DEFINE_string('test_dataset_name', 'imgs/00000',
                           """Where the test sequences are.""")
tf.app.flags.DEFINE_string('img_type', 'png',
                           """Image types.""")
tf.app.flags.DEFINE_float('test_ratio', 1.0,
                          """Rescaling factor for the testing sequence.""")
tf.app.flags.DEFINE_string('ckpt_path', 'temp_online_ckpt/model.ckpt-1000',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_string('output_dir', 'output',
                           """Output folder.""")
tf.app.flags.DEFINE_string('GPU_ID', '',
                           """GPU ID""")

GPU_ID = FLAGS.GPU_ID

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
import sys
sys.path.insert(1, 'tfoptflow/tfoptflow/')
from copy import deepcopy
from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
nn_opts['verbose'] = True
nn_opts['ckpt_path'] = 'tfoptflow/tfoptflow/models/pwcnet-lg-6-2-multisteps-chairsthingsmix/pwcnet.ckpt-595000'
nn_opts['batch_size'] = 1
nn_opts['gpu_devices'] = ['/device:CPU:0']
nn_opts['controller'] = '/device:CPU:0'
nn_opts['use_dense_cx'] = True
nn_opts['use_res_cx'] = True
nn_opts['pyr_lvls'] = 6
nn_opts['flow_pred_lvl'] = 2


I0 = cv2.imread(FLAGS.test_dataset_name+'_I0.'+FLAGS.img_type).astype(np.float32)[..., ::-1] / 255.0
I1 = cv2.imread(FLAGS.test_dataset_name+'_I1.'+FLAGS.img_type).astype(np.float32)[..., ::-1] / 255.0
I2 = cv2.imread(FLAGS.test_dataset_name+'_I2.'+FLAGS.img_type).astype(np.float32)[..., ::-1] / 255.0
I3 = cv2.imread(FLAGS.test_dataset_name+'_I3.'+FLAGS.img_type).astype(np.float32)[..., ::-1] / 255.0
I4 = cv2.imread(FLAGS.test_dataset_name+'_I4.'+FLAGS.img_type).astype(np.float32)[..., ::-1] / 255.0
ORIGINAL_H = I0.shape[0]
ORIGINAL_W = I0.shape[1]

RESIZED_H = int(np.ceil(float(ORIGINAL_H) * FLAGS.test_ratio / 16.0))*16
RESIZED_W = int(np.ceil(float(ORIGINAL_W) * FLAGS.test_ratio / 16.0))*16
print(RESIZED_H)
print(RESIZED_W)

I0 = np.expand_dims(cv2.resize(I0, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC), 0)
I1 = np.expand_dims(cv2.resize(I1, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC), 0)
I2 = np.expand_dims(cv2.resize(I2, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC), 0)
I3 = np.expand_dims(cv2.resize(I3, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC), 0)
I4 = np.expand_dims(cv2.resize(I4, dsize=(RESIZED_W, RESIZED_H), interpolation=cv2.INTER_CUBIC), 0)



CROP_PATCH_H = RESIZED_H
CROP_PATCH_W = RESIZED_W

def flow_to_img(flow):
    flow_magnitude = tf.sqrt(1e-6 + flow[..., 0]**2.0 + flow[..., 1]**2.0)
    flow_angle = tf.atan2(flow[..., 0], flow[..., 1])

    hsv_0 = ((flow_angle / np.pi)+1.0)/2.0
    hsv_1 = (flow_magnitude - tf.reduce_min(flow_magnitude, axis=[1, 2], keepdims=True)) / (1e-6 + tf.reduce_max(flow_magnitude, axis=[1, 2], keepdims=True) - tf.reduce_min(flow_magnitude, axis=[1, 2], keepdims=True))
    hsv_2 = tf.ones(tf.shape(hsv_0))
    hsv = tf.stack([hsv_0, hsv_1, hsv_2], -1)
    rgb = tf.image.hsv_to_rgb(hsv)

    return rgb

def warp(I, F):
    return tf.reshape(dense_image_warp(I, tf.stack([-F[..., 1], -F[..., 0]], -1)), [FLAGS.batch_size, CROP_PATCH_H, CROP_PATCH_W, 3])



with tf.Graph().as_default():

    fused_frame0 = tf.placeholder(tf.float32, [1, CROP_PATCH_H, CROP_PATCH_W, 3])
    fused_frame1 = tf.placeholder(tf.float32, [1, CROP_PATCH_H, CROP_PATCH_W, 3])
    fused_frame2 = tf.placeholder(tf.float32, [1, CROP_PATCH_H, CROP_PATCH_W, 3])
    fused_frame3 = tf.placeholder(tf.float32, [1, CROP_PATCH_H, CROP_PATCH_W, 3])
    fused_frame4 = tf.placeholder(tf.float32, [1, CROP_PATCH_H, CROP_PATCH_W, 3])

    fused_frame0_small = tf.image.resize_bilinear(fused_frame0, [192, 320])
    fused_frame1_small = tf.image.resize_bilinear(fused_frame1, [192, 320])
    fused_frame2_small = tf.image.resize_bilinear(fused_frame2, [192, 320])
    fused_frame3_small = tf.image.resize_bilinear(fused_frame3, [192, 320])
    fused_frame4_small = tf.image.resize_bilinear(fused_frame4, [192, 320])
    
    def PWC_full(F0, F1, F2, F3, F4, B0, B1, B2, B3, B4, lvl_h, lvl_w, pwc_h, pwc_w):
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
        ratio_tensor = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(np.asarray([ratio_w, ratio_h]), dtype=tf.float32), 0), 0), 0)

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


    model = Decomposition_Net_Translation(CROP_PATCH_H//16, CROP_PATCH_W//16, False, False, False)
    FF01, FF02, FF03, FF04, \
    FF10, FF12, FF13, FF14, \
    FF20, FF21, FF23, FF24, \
    FF30, FF31, FF32, FF34, \
    FF40, FF41, FF42, FF43, \
    FB01, FB02, FB03, FB04, \
    FB10, FB12, FB13, FB14, \
    FB20, FB21, FB23, FB24, \
    FB30, FB31, FB32, FB34, \
    FB40, FB41, FB42, FB43 = model.inference(fused_frame0, fused_frame1, fused_frame2, fused_frame3, fused_frame4)
    
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
                                                                                FF01, FF02, FF03, FF04,
                                                                                FF10, FF12, FF13, FF14,
                                                                                FF20, FF21, FF23, FF24,
                                                                                FF30, FF31, FF32, FF34,
                                                                                FF40, FF41, FF42, FF43,
                                                                                FB01, FB02, FB03, FB04,
                                                                                FB10, FB12, FB13, FB14,
                                                                                FB20, FB21, FB23, FB24,
                                                                                FB30, FB31, FB32, FB34,
                                                                                FB40, FB41, FB42, FB43)
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
    FF01, FF02, FF03, FF04, \
    FF10, FF12, FF13, FF14, \
    FF20, FF21, FF23, FF24, \
    FF30, FF31, FF32, FF34, \
    FF40, FF41, FF42, FF43, \
    FB01, FB02, FB03, FB04, \
    FB10, FB12, FB13, FB14, \
    FB20, FB21, FB23, FB24, \
    FB30, FB31, FB32, FB34, \
    FB40, FB41, FB42, FB43 = PWC_full(F0_pred_4_up, F1_pred_4_up, F2_pred_4_up, F3_pred_4_up, F4_pred_4_up,
                                      B0_pred_4_up, B1_pred_4_up, B2_pred_4_up, B3_pred_4_up, B4_pred_4_up,
                                      CROP_PATCH_H // (2 ** 3), CROP_PATCH_W // (2 ** 3),
                                      int(np.ceil(float(CROP_PATCH_H // (2 ** 3)) / 64.0)) * 64,
                                      int(np.ceil(float(CROP_PATCH_W // (2 ** 3)) / 64.0)) * 64)

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
                                                                                FF01, FF02, FF03, FF04,
                                                                                FF10, FF12, FF13, FF14,
                                                                                FF20, FF21, FF23, FF24,
                                                                                FF30, FF31, FF32, FF34,
                                                                                FF40, FF41, FF42, FF43,
                                                                                FB01, FB02, FB03, FB04,
                                                                                FB10, FB12, FB13, FB14,
                                                                                FB20, FB21, FB23, FB24,
                                                                                FB30, FB31, FB32, FB34,
                                                                                FB40, FB41, FB42, FB43)
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
    FF01, FF02, FF03, FF04, \
    FF10, FF12, FF13, FF14, \
    FF20, FF21, FF23, FF24, \
    FF30, FF31, FF32, FF34, \
    FF40, FF41, FF42, FF43, \
    FB01, FB02, FB03, FB04, \
    FB10, FB12, FB13, FB14, \
    FB20, FB21, FB23, FB24, \
    FB30, FB31, FB32, FB34, \
    FB40, FB41, FB42, FB43 = PWC_full(F0_pred_3_up, F1_pred_3_up, F2_pred_3_up, F3_pred_3_up, F4_pred_3_up,
                                      B0_pred_3_up, B1_pred_3_up, B2_pred_3_up, B3_pred_3_up, B4_pred_3_up,
                                      CROP_PATCH_H // (2 ** 2), CROP_PATCH_W // (2 ** 2),
                                      int(np.ceil(float(CROP_PATCH_H // (2 ** 2)) / 64.0)) * 64,
                                      int(np.ceil(float(CROP_PATCH_W // (2 ** 2)) / 64.0)) * 64)
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
                                                                                FF01, FF02, FF03, FF04,
                                                                                FF10, FF12, FF13, FF14,
                                                                                FF20, FF21, FF23, FF24,
                                                                                FF30, FF31, FF32, FF34,
                                                                                FF40, FF41, FF42, FF43,
                                                                                FB01, FB02, FB03, FB04,
                                                                                FB10, FB12, FB13, FB14,
                                                                                FB20, FB21, FB23, FB24,
                                                                                FB30, FB31, FB32, FB34,
                                                                                FB40, FB41, FB42, FB43)
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
    FF01, FF02, FF03, FF04, \
    FF10, FF12, FF13, FF14, \
    FF20, FF21, FF23, FF24, \
    FF30, FF31, FF32, FF34, \
    FF40, FF41, FF42, FF43, \
    FB01, FB02, FB03, FB04, \
    FB10, FB12, FB13, FB14, \
    FB20, FB21, FB23, FB24, \
    FB30, FB31, FB32, FB34, \
    FB40, FB41, FB42, FB43 = PWC_full(F0_pred_2_up, F1_pred_2_up, F2_pred_2_up, F3_pred_2_up, F4_pred_2_up,
                                      B0_pred_2_up, B1_pred_2_up, B2_pred_2_up, B3_pred_2_up, B4_pred_2_up,
                                      CROP_PATCH_H // (2 ** 1), CROP_PATCH_W // (2 ** 1),
                                      int(np.ceil(float(CROP_PATCH_H // (2 ** 1)) / 64.0)) * 64,
                                      int(np.ceil(float(CROP_PATCH_W // (2 ** 1)) / 64.0)) * 64)

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
                                                                                FF01, FF02, FF03, FF04,
                                                                                FF10, FF12, FF13, FF14,
                                                                                FF20, FF21, FF23, FF24,
                                                                                FF30, FF31, FF32, FF34,
                                                                                FF40, FF41, FF42, FF43,
                                                                                FB01, FB02, FB03, FB04,
                                                                                FB10, FB12, FB13, FB14,
                                                                                FB20, FB21, FB23, FB24,
                                                                                FB30, FB31, FB32, FB34,
                                                                                FB40, FB41, FB42, FB43)
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
    FF01, FF02, FF03, FF04, \
    FF10, FF12, FF13, FF14, \
    FF20, FF21, FF23, FF24, \
    FF30, FF31, FF32, FF34, \
    FF40, FF41, FF42, FF43, \
    FB01, FB02, FB03, FB04, \
    FB10, FB12, FB13, FB14, \
    FB20, FB21, FB23, FB24, \
    FB30, FB31, FB32, FB34, \
    FB40, FB41, FB42, FB43 = PWC_full(F0_pred_1_up, F1_pred_1_up, F2_pred_1_up, F3_pred_1_up, F4_pred_1_up,
                                      B0_pred_1_up, B1_pred_1_up, B2_pred_1_up, B3_pred_1_up, B4_pred_1_up,
                                      CROP_PATCH_H // (2 ** 0), CROP_PATCH_W // (2 ** 0),
                                      int(np.ceil(float(CROP_PATCH_H // (2 ** 0)) / 64.0)) * 64,
                                      int(np.ceil(float(CROP_PATCH_W // (2 ** 0)) / 64.0)) * 64)

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
                                                                                FF01, FF02, FF03, FF04,
                                                                                FF10, FF12, FF13, FF14,
                                                                                FF20, FF21, FF23, FF24,
                                                                                FF30, FF31, FF32, FF34,
                                                                                FF40, FF41, FF42, FF43,
                                                                                FB01, FB02, FB03, FB04,
                                                                                FB10, FB12, FB13, FB14,
                                                                                FB20, FB21, FB23, FB24,
                                                                                FB30, FB31, FB32, FB34,
                                                                                FB40, FB41, FB42, FB43)

    sess = tf.Session()
    
    saver2 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "pwcnet" in v.name])
    saver2.restore(sess, nn_opts['ckpt_path'])
    saver4 = tf.train.Saver(var_list=[v for v in tf.all_variables() if "pwcnet" not in v.name])
    saver4.restore(sess, FLAGS.ckpt_path)
    
    
    import cv2

    out_path = FLAGS.output_dir + '/'
    F2, B2 = sess.run([F2_pred_0, B2_pred_0], feed_dict={fused_frame0: I0, fused_frame1: I1, fused_frame2: I2, fused_frame3: I3, fused_frame4: I4})
    
    cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'F2_norm.png', np.clip(np.round(cv2.resize(F2[0, :, :, ::-1]/np.max(F2[0, :, :, ::-1]), dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'B2_norm.png', np.clip(np.round(cv2.resize(B2[0, :, :, ::-1]/np.max(B2[0, :, :, ::-1]), dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'F2.png', np.clip(np.round(cv2.resize(F2[0, :, :, ::-1], dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
    cv2.imwrite(out_path + FLAGS.test_dataset_name[-5:]+'B2.png', np.clip(np.round(cv2.resize(B2[0, :, :, ::-1], dsize=(ORIGINAL_W, ORIGINAL_H), interpolation=cv2.INTER_CUBIC) * 255.0), 0.0, 255.0).astype(np.uint8))
