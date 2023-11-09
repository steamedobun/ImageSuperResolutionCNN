"""
初始化
"""
import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras

# 尝试tensorflow-gpu2.4.0版本为from tensorflow.keras.utils导库
from keras import layers
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array

# from tensorflow.python.keras.utils.preprocessing import image_dataset_from_directory
# tensorflow-gpu2.10.0版本更改为tf.keras.utils.image_dataset_from_directory调用方法

from IPython.display import display

# 导入 plt库 pillow库
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import PIL
from PIL import Image

"""
ESPCN (Efficient Sub-Pixel CNN), proposed by [Shi, 2016](https://arxiv.org/abs/1609.05158)
原网络结构 Efficient Sub-Pixel CNN
[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html).
使用数据集 BSDS500 dataset
"""

# 读取C:\Users\******\.keras\datasets下的数据集(官网下载)
# dataset_url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
# data_dir = keras.utils.get_file(origin=dataset_url, fname="BSR", untar=True)
# root_dir = os.path.join(data_dir, "BSDS500/data")

# 读取当前项目文件夹下的数据集
data_dir = "./datasets/BSR"
print(data_dir)
root_dir = os.path.splitext(data_dir)[0] + "/BSDS500/data"
print(root_dir)

"""
training datasets   训练集
validation datasets 验证集
"""

# 裁剪后图像大小
crop_size = 300
# 上采样倍数
upscale_factor = 3
# 模型输入大小
input_size = crop_size // upscale_factor
# 批次大小
batch_size = 8

# tf.keras.utils.image_dataset_from_directory()函数生成训练集和验证集
# 训练集
train_ds = tf.keras.utils.image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

# 验证集
valid_ds = tf.keras.utils.image_dataset_from_directory(
    root_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

"""
归一化
"""


# 归一化 将输入图像像素值范围从(0, 255)归一化到(0, 1)
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# 每一张图像像素范围从(0, 255)归一化到(0, 1)
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)

'''
# train_ds.take(1)返回数据集train_ds中的第一个批次数据
for batch in train_ds.take(1):
    for img in batch:
        # 将一个张量转换成PIL图像格式，方便展示
        # display(array_to_img(img))在Jupyter Notebook中展示该张图像
        display(array_to_img(img))
'''

"""
我们准备了一个测试图像路径的数据集，我们将在此示例结束时用于视觉评估。
"""

# 加载整个数据集位置
dataset = os.path.join(root_dir, "images")
# 测试集目录
test_path = os.path.join(dataset, "test")

# 通过列表推导式获取测试集中所有以.jpg结尾的图像的路径列表，并使用sorted()函数将路径排序，以保证读入图像的顺序不变
test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpg")
    ]
)

"""
RGB转换为YUV
[YUV colour space](https://en.wikipedia.org/wiki/YUV).

对于输入的低分辨率图像，我们会对其进行剪裁，提取Y通道的亮度信息，
然后使用面积方法重新调整图像大小（如果使用了PIL库，则使用双三次插值方法）。
我们只考虑YUV颜色空间中的亮度通道，因为人类对亮度变化更敏感。

对于目标数据（高分辨率图像），我们只进行图像剪裁和提取Y通道的亮度信息的处理。

这些步骤的目的是为了将原始的彩色图像转变成对于模型来说更好学习的Y通道图像。
同时，通过对图像进行插值操作，可以将其大小缩小到模型输入大小，
使模型可以更容易地学习和处理这些数据
"""


# tensorflow 操作（也可以是用PIL双三次插值方法）
def process_input(input, input_size, upscale_factor):
    # 转换为YUV格式(原理theory)
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    # 分为YUV三通道
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    # 插值操作，使其大小为input_size * input_size（只取Y通道）
    return tf.image.resize(y, [input_size, input_size], method="area")


def process_target(input):
    # 转换为YUV格式(原理theory)
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    # 只取亮度（Y）通道
    return y


# 对训练集应用process_input和process_target函数
# 将每张原始图像转换为下采样的YUV格式的输入图像和原始YUV格式的目标图像的元组
train_ds = train_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
# 训练集缓存到内存中 加速数据读取
train_ds = train_ds.prefetch(buffer_size=32)

# 对验证集应用process_input和process_target函数
# 将每张原始图像转换为下采样的YUV格式的输入图像和原始YUV格式的目标图像的元组
valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_size, upscale_factor), process_target(x))
)
# 将验证集缓存到内存中 加速数据读取
valid_ds = valid_ds.prefetch(buffer_size=32)

"""
Let's take a look at the input and target data.
"""

# 观察输入和目标数据图像
for batch in train_ds.take(1):
    for img in batch[0]:
        display(array_to_img(img))
    for img in batch[1]:
        display(array_to_img(img))

"""
## Build a model

Compared to the paper, we add one more layer and we use the `relu` activation function
instead of `tanh`.
It achieves better performance even though we train the model for fewer epochs.
"""

"""
def get_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
"""


def get_model(upscale_factor=3, channels=1):
    inputs = keras.Input(shape=(None, None, channels))

    # 前置卷积块
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='Orthogonal')(inputs)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='Orthogonal')(x)

    # 残差块
    # 每个残差单元的输入先经过一层卷积，再经过ReLU激活函数，然后再通过另一层卷积得到残差
    # 可以加深网络深度，同时避免训练过程中的梯度消失和梯度爆炸问题
    for i in range(3):
        residual = x
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='Orthogonal')(x)
        x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                          kernel_initializer='Orthogonal')(x)
        x = layers.add([x, residual])

    # 后置卷积块
    x = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_initializer='Orthogonal')(x)
    x = layers.Conv2D(filters=channels * (upscale_factor ** 2), kernel_size=3, strides=1, padding='same',
                      activation='relu', kernel_initializer='Orthogonal')(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)


"""
## Define utility functions

We need to define several utility functions to monitor our results:

- `plot_results` to plot an save an image.
- `get_lowres_image` to convert an image to its low-resolution version.
- `upscale_image` to turn a low-resolution image to
a high-resolution version reconstructed by the model.
In this function, we use the `y` channel from the YUV color space
as input to the model and then combine the output with the
other channels to obtain an RGB image.
"""


# 画图展示
def plot_results(img, prefix, title):
    """Plot the result with zoom-in area."""
    # 将图像转化为浮点数类型，并将其标准化到[0,1]范围内
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # Create a new figure with a default 111 subplot.
    # 创建一个新的Matplotlib图形，包括一个子图和对应的代理图像对象
    fig, ax = plt.subplots()
    # origin="lower"指定坐标系原点的位置为图像的左下角
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.title(title)
    # zoom-factor: 2.0, location: upper-left
    # 放大展示细节分辨率
    axins = zoomed_inset_axes(ax, 2, loc=2)
    axins.imshow(img_array[::-1], origin="lower")

    # Specify the limits.
    x1, x2, y1, y2 = 200, 300, 100, 200
    # Apply the x-limits.
    axins.set_xlim(x1, x2)
    # Apply the y-limits.
    axins.set_ylim(y1, y2)

    plt.yticks(visible=False)
    plt.xticks(visible=False)

    # Make the line.
    # 模拟放大图片的线条
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="red")
    plt.savefig("./image_train/image_prediction" + str(prefix) + "-" + title + ".png")
    plt.show()


# 获取低分辨率图片
def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )


# 低分辨率图像输入预测(模型预测)
def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    # 分离YBR中的亮度通道（Y）单独处理
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    # 将数据的形状从(H, W)扩展为(1, H, W, 1)
    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    # 将Numpy数组out_img_y的值还原为颜色值
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img


"""
The `ESPCNCallback` object will compute and display
the [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) metric.
This is the main metric we use to evaluate super-resolution performance.
"""


class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))


"""
Define `ModelCheckpoint` and `EarlyStopping` callbacks.
"""
# EarlyStopping回调用于在训练过程中监控某些指标，并在未能取得验证集性能改善时及早停止训练
# 避免模型在过拟合的情况下继续训练
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

checkpoint_filepath = "./model_checkpoint/"

# 检查点定期保存模型的权重
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

# 网络模型
model = get_model(upscale_factor=upscale_factor, channels=1)

# 打印网络结构
model.summary()

# 回调处理函数
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]

# MeanSquaredError()损失函数
# 计算每一对预测值和目标值之间的均方差，并返回其平均值
loss_fn = keras.losses.MeanSquaredError()

# Adam优化器
# 基于梯度算法可以有效优化神经网络模型参数
# 学习率 learning_rate=0.001
optimizer = keras.optimizers.Adam(learning_rate=0.001)

"""
训练模型
"""
# 训练遍数
epochs = 200

# 编译模型
model.compile(
    optimizer=optimizer,
    loss=loss_fn,
)

# 训练模型(verbose 0 不输出日志信息 1 输出进度条 2 输出每个epoch的训练和验证信息)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)

# The model weights (that are considered the best) are loaded into the model.
# 权重保存
model.load_weights(checkpoint_filepath)

# 保存模型
model.save("./model/")

"""
预测 测试
让我们计算一些图像的重构版本并保存结果
峰值信噪比( PSNR ) 是一个工程术语，表示信号的最大可能功率与影响其表示保真度的破坏噪声功率之间的比率。
由于许多信号具有非常宽的动态范围，因此 PSNR 通常使用分贝标度表示为对数。

PSNR 最常用于测量有损压缩编解码器（例如，图像压缩）的重建质量。
这种情况下的信号是原始数据，噪声是压缩引入的误差。
在比较压缩编解码器时，PSNR 是人类对重建质量感知的 近似值。
"""
# 评估模型性能
total_bicubic_psnr = 0.0
total_test_psnr = 0.0

# 测试集中选择前10张图片进行评估
for index, test_img_path in enumerate(test_img_paths[60:70]):
    img = load_img(test_img_path)
    lowres_input = get_lowres_image(img, upscale_factor)
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input)
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

    print(
        "PSNR of low resolution image and high resolution image is %.4f" % bicubic_psnr
    )
    print("PSNR of predict and high resolution is %.4f" % test_psnr)
    plot_results(lowres_img, index, "lowres")
    plot_results(highres_img, index, "highres")
    plot_results(prediction, index, "prediction")

print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))


"""
PSNR of low resolution image and high resolution image is 24.7959
PSNR of predict and high resolution is 28.3344

PSNR (Peak Signal-to-Noise Ratio)是一种评估图像质量的度量标准，通常用于图像复原和压缩的评估.
PSNR值越高，表示预测的图片和原图之间的误差越小，图像质量越好。

对于第一句话，PSNR的值为24.7959，说明低分辨率图像和高分辨率图像之间的误差较大，图像质量较差。
而第二句话中，PSNR的值为28.3344，说明预测的图像与高分辨率原图之间的误差较小，图像质量较好。
"""