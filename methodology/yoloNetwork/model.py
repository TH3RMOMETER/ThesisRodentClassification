#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    18-May-2023 16:52:04

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_model(shape=[37648, 52, 1]):
    input_1_unnormalized = keras.Input(shape=shape, name="input_1_unnormalized")
    input_1 = SubtractConstantLayer(shape, name="input_1_")(input_1_unnormalized)
    Conv1 = layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same", name="Conv1_")(input_1)
    bn_Conv1 = layers.BatchNormalization(epsilon=0.001000, name="bn_Conv1_")(Conv1)
    Conv1_relu = layers.ReLU(max_value=6.000000)(bn_Conv1)
    expanded_conv_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                                     name="expanded_conv_depthwise_")(Conv1_relu)
    expanded_conv_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="expanded_conv_depthwise_BN_")(
        expanded_conv_depthwise)
    expanded_conv_depthwise_relu = layers.ReLU(max_value=6.000000)(expanded_conv_depthwise_BN)
    expanded_conv_project = layers.Conv2D(16, (1, 1), padding="same", name="expanded_conv_project_")(
        expanded_conv_depthwise_relu)
    expanded_conv_project_BN = layers.BatchNormalization(epsilon=0.001000, name="expanded_conv_project_BN_")(
        expanded_conv_project)
    block_1_expand = layers.Conv2D(96, (1, 1), padding="same", name="block_1_expand_")(expanded_conv_project_BN)
    block_1_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_1_expand_BN_")(block_1_expand)
    block_1_expand_relu = layers.ReLU(max_value=6.000000)(block_1_expand_BN)
    block_1_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", strides=(2, 2), use_bias=True,
                                               name="block_1_depthwise_")(block_1_expand_relu)
    block_1_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_1_depthwise_BN_")(block_1_depthwise)
    block_1_depthwise_relu = layers.ReLU(max_value=6.000000)(block_1_depthwise_BN)
    block_1_project = layers.Conv2D(24, (1, 1), padding="same", name="block_1_project_")(block_1_depthwise_relu)
    block_1_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_1_project_BN_")(block_1_project)
    block_2_expand = layers.Conv2D(144, (1, 1), padding="same", name="block_2_expand_")(block_1_project_BN)
    block_2_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_2_expand_BN_")(block_2_expand)
    block_2_expand_relu = layers.ReLU(max_value=6.000000)(block_2_expand_BN)
    block_2_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                               name="block_2_depthwise_")(block_2_expand_relu)
    block_2_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_2_depthwise_BN_")(block_2_depthwise)
    block_2_depthwise_relu = layers.ReLU(max_value=6.000000)(block_2_depthwise_BN)
    block_2_project = layers.Conv2D(24, (1, 1), padding="same", name="block_2_project_")(block_2_depthwise_relu)
    block_2_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_2_project_BN_")(block_2_project)
    block_2_add = layers.Add()([block_2_project_BN, block_1_project_BN])
    block_3_expand = layers.Conv2D(144, (1, 1), padding="same", name="block_3_expand_")(block_2_add)
    block_3_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_3_expand_BN_")(block_3_expand)
    block_3_expand_relu = layers.ReLU(max_value=6.000000)(block_3_expand_BN)
    block_3_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", strides=(2, 2), use_bias=True,
                                               name="block_3_depthwise_")(block_3_expand_relu)
    block_3_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_3_depthwise_BN_")(block_3_depthwise)
    block_3_depthwise_relu = layers.ReLU(max_value=6.000000)(block_3_depthwise_BN)
    block_3_project = layers.Conv2D(32, (1, 1), padding="same", name="block_3_project_")(block_3_depthwise_relu)
    block_3_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_3_project_BN_")(block_3_project)
    block_4_expand = layers.Conv2D(192, (1, 1), padding="same", name="block_4_expand_")(block_3_project_BN)
    block_4_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_4_expand_BN_")(block_4_expand)
    block_4_expand_relu = layers.ReLU(max_value=6.000000)(block_4_expand_BN)
    block_4_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                               name="block_4_depthwise_")(block_4_expand_relu)
    block_4_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_4_depthwise_BN_")(block_4_depthwise)
    block_4_depthwise_relu = layers.ReLU(max_value=6.000000)(block_4_depthwise_BN)
    block_4_project = layers.Conv2D(32, (1, 1), padding="same", name="block_4_project_")(block_4_depthwise_relu)
    block_4_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_4_project_BN_")(block_4_project)
    block_4_add = layers.Add()([block_4_project_BN, block_3_project_BN])
    block_5_expand = layers.Conv2D(192, (1, 1), padding="same", name="block_5_expand_")(block_4_add)
    block_5_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_5_expand_BN_")(block_5_expand)
    block_5_expand_relu = layers.ReLU(max_value=6.000000)(block_5_expand_BN)
    block_5_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                               name="block_5_depthwise_")(block_5_expand_relu)
    block_5_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_5_depthwise_BN_")(block_5_depthwise)
    block_5_depthwise_relu = layers.ReLU(max_value=6.000000)(block_5_depthwise_BN)
    block_5_project = layers.Conv2D(32, (1, 1), padding="same", name="block_5_project_")(block_5_depthwise_relu)
    block_5_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_5_project_BN_")(block_5_project)
    block_5_add = layers.Add()([block_5_project_BN, block_4_add])
    block_6_expand = layers.Conv2D(192, (1, 1), padding="same", name="block_6_expand_")(block_5_add)
    block_6_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_6_expand_BN_")(block_6_expand)
    block_6_expand_relu = layers.ReLU(max_value=6.000000)(block_6_expand_BN)
    block_6_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", strides=(2, 2), use_bias=True,
                                               name="block_6_depthwise_")(block_6_expand_relu)
    block_6_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_6_depthwise_BN_")(block_6_depthwise)
    block_6_depthwise_relu = layers.ReLU(max_value=6.000000)(block_6_depthwise_BN)
    block_6_project = layers.Conv2D(64, (1, 1), padding="same", name="block_6_project_")(block_6_depthwise_relu)
    block_6_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_6_project_BN_")(block_6_project)
    block_7_expand = layers.Conv2D(384, (1, 1), padding="same", name="block_7_expand_")(block_6_project_BN)
    block_7_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_7_expand_BN_")(block_7_expand)
    block_7_expand_relu = layers.ReLU(max_value=6.000000)(block_7_expand_BN)
    block_7_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                               name="block_7_depthwise_")(block_7_expand_relu)
    block_7_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_7_depthwise_BN_")(block_7_depthwise)
    block_7_depthwise_relu = layers.ReLU(max_value=6.000000)(block_7_depthwise_BN)
    block_7_project = layers.Conv2D(64, (1, 1), padding="same", name="block_7_project_")(block_7_depthwise_relu)
    block_7_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_7_project_BN_")(block_7_project)
    block_7_add = layers.Add()([block_7_project_BN, block_6_project_BN])
    block_8_expand = layers.Conv2D(384, (1, 1), padding="same", name="block_8_expand_")(block_7_add)
    block_8_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_8_expand_BN_")(block_8_expand)
    block_8_expand_relu = layers.ReLU(max_value=6.000000)(block_8_expand_BN)
    block_8_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                               name="block_8_depthwise_")(block_8_expand_relu)
    block_8_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_8_depthwise_BN_")(block_8_depthwise)
    block_8_depthwise_relu = layers.ReLU(max_value=6.000000)(block_8_depthwise_BN)
    block_8_project = layers.Conv2D(64, (1, 1), padding="same", name="block_8_project_")(block_8_depthwise_relu)
    block_8_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_8_project_BN_")(block_8_project)
    block_8_add = layers.Add()([block_8_project_BN, block_7_add])
    block_9_expand = layers.Conv2D(384, (1, 1), padding="same", name="block_9_expand_")(block_8_add)
    block_9_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_9_expand_BN_")(block_9_expand)
    block_9_expand_relu = layers.ReLU(max_value=6.000000)(block_9_expand_BN)
    block_9_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                               name="block_9_depthwise_")(block_9_expand_relu)
    block_9_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_9_depthwise_BN_")(block_9_depthwise)
    block_9_depthwise_relu = layers.ReLU(max_value=6.000000)(block_9_depthwise_BN)
    block_9_project = layers.Conv2D(64, (1, 1), padding="same", name="block_9_project_")(block_9_depthwise_relu)
    block_9_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_9_project_BN_")(block_9_project)
    block_9_add = layers.Add()([block_9_project_BN, block_8_add])
    block_10_expand = layers.Conv2D(384, (1, 1), padding="same", name="block_10_expand_")(block_9_add)
    block_10_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_10_expand_BN_")(block_10_expand)
    block_10_expand_relu = layers.ReLU(max_value=6.000000)(block_10_expand_BN)
    block_10_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                                name="block_10_depthwise_")(block_10_expand_relu)
    block_10_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_10_depthwise_BN_")(
        block_10_depthwise)
    block_10_depthwise_relu = layers.ReLU(max_value=6.000000)(block_10_depthwise_BN)
    block_10_project = layers.Conv2D(96, (1, 1), padding="same", name="block_10_project_")(block_10_depthwise_relu)
    block_10_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_10_project_BN_")(block_10_project)
    block_11_expand = layers.Conv2D(576, (1, 1), padding="same", name="block_11_expand_")(block_10_project_BN)
    block_11_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_11_expand_BN_")(block_11_expand)
    block_11_expand_relu = layers.ReLU(max_value=6.000000)(block_11_expand_BN)
    block_11_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                                name="block_11_depthwise_")(block_11_expand_relu)
    block_11_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_11_depthwise_BN_")(
        block_11_depthwise)
    block_11_depthwise_relu = layers.ReLU(max_value=6.000000)(block_11_depthwise_BN)
    block_11_project = layers.Conv2D(96, (1, 1), padding="same", name="block_11_project_")(block_11_depthwise_relu)
    block_11_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_11_project_BN_")(block_11_project)
    block_11_add = layers.Add()([block_11_project_BN, block_10_project_BN])
    block_12_expand = layers.Conv2D(576, (1, 1), padding="same", name="block_12_expand_")(block_11_add)
    block_12_expand_BN = layers.BatchNormalization(epsilon=0.001000, name="block_12_expand_BN_")(block_12_expand)
    block_12_expand_relu = layers.ReLU(max_value=6.000000)(block_12_expand_BN)
    block_12_depthwise = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same", use_bias=True,
                                                name="block_12_depthwise_")(block_12_expand_relu)
    block_12_depthwise_BN = layers.BatchNormalization(epsilon=0.001000, name="block_12_depthwise_BN_")(
        block_12_depthwise)
    block_12_depthwise_relu = layers.ReLU(max_value=6.000000)(block_12_depthwise_BN)
    block_12_project = layers.Conv2D(96, (1, 1), padding="same", name="block_12_project_")(block_12_depthwise_relu)
    block_12_project_BN = layers.BatchNormalization(epsilon=0.001000, name="block_12_project_BN_")(block_12_project)
    block_12_add = layers.Add()([block_12_project_BN, block_11_add])
    yolov2Conv1 = layers.Conv2D(96, (3, 3), padding="same", name="yolov2Conv1_")(block_12_add)
    yolov2Batch1 = layers.BatchNormalization(epsilon=0.000010, name="yolov2Batch1_")(yolov2Conv1)
    yolov2Relu1 = layers.ReLU()(yolov2Batch1)
    yolov2Conv2 = layers.Conv2D(96, (3, 3), padding="same", name="yolov2Conv2_")(yolov2Relu1)
    yolov2Batch2 = layers.BatchNormalization(epsilon=0.000010, name="yolov2Batch2_")(yolov2Conv2)
    yolov2Relu2 = layers.ReLU()(yolov2Batch2)
    yolov2ClassConv = layers.Conv2D(48, (1, 1), name="yolov2ClassConv_")(yolov2Relu2)
    maxPool = layers.MaxPooling2D()(yolov2ClassConv)
    flatten = layers.Flatten()(maxPool)
    binaryClassDense = layers.Dense(1, activation="sigmoid", name="binaryClassDense_")(flatten)
    model = keras.Model(inputs=[input_1_unnormalized], outputs=[binaryClassDense])
    return model


## Helper layers:

class SubtractConstantLayer(tf.keras.layers.Layer):
    def __init__(self, shape, name=None):
        super(SubtractConstantLayer, self).__init__(name=name)
        self.const = tf.Variable(initial_value=tf.zeros(shape), trainable=False)

    def call(self, input):
        return input - self.const
