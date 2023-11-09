import os
import tensorflow as tf
import keras
from keras import layers
from keras.utils import load_img
from keras.utils import img_to_array

import numpy as np
import matplotlib.pyplot as plt
import PIL


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


# 获取低分辨率图片
def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )


# 低分辨率图像上采样
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


def save_plot_results(img, prefix, title, save_path):
    # 将图像转化为浮点数类型，并将其标准化到[0,1]范围内
    img_array = img_to_array(img)
    img_array = img_array.astype("float32") / 255.0

    # 创建一个新的Matplotlib图形，包括一个子图和对应的代理图像对象
    fig, ax = plt.subplots()

    # origin="lower"指定坐标系原点的位置为图像的左下角
    im = ax.imshow(img_array[::-1], origin="lower")

    plt.axis("off")

    # 保存图像
    save_name = str(index) + "-" + title + ".png"
    save_path_res = save_path + title + "/" + save_name
    plt.savefig(save_path_res, bbox_inches="tight", pad_inches=0)


if __name__ == '__main__':
    """
    # 读取当前项目文件夹下的数据集
    root_dir = "./datasets/BSR/BSDS500/data"
    
    # 整个数据集位置
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

    save_path = "./image_main/"

    # 上采样倍数
    upscale_factor = 3

    # 网络模型
    model = get_model(upscale_factor=upscale_factor, channels=1)
    # 打印网络结构
    model.summary()
    # MeanSquaredError()损失函数 计算每一对预测值和目标值之间的均方差，并返回其平均值
    loss_fn = keras.losses.MeanSquaredError()
    # Adam优化器 基于梯度算法可以有效优化神经网络模型参数
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    # 编译模型
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
    )
    # 加载权重
    checkpoint_filepath = "./model_checkpoint/"
    model.load_weights(checkpoint_filepath)

    # 输入图片位置
    input_path = "./image_main/input"
    input_img_paths = sorted(
        [
            os.path.join(input_path, fname)
            for fname in os.listdir(input_path)
            if fname.endswith(".jpg")
        ]
    )

    # image_main中图片进行分辨率提升
    for index, input_img_path in enumerate(input_img_paths):
        # 原始图像
        img = load_img(input_img_path)

        # 改为低分辨率图像(真正效果实现，低分辨率为输入图像，然后对其操作)
        lowres_input = get_lowres_image(img, upscale_factor)

        w = lowres_input.size[0] * upscale_factor
        h = lowres_input.size[1] * upscale_factor

        # 经过CNN网络预测
        prediction = upscale_image(model, lowres_input)

        lowres_img = lowres_input.resize((w, h))

        # 图像转为数组，一般用于计算
        # lowres_img_arr = img_to_array(lowres_img)
        # predict_img_arr = img_to_array(prediction)
        
        save_plot_results(lowres_img, index, "lowers", save_path=save_path)
        save_plot_results(prediction, index, "prediction", save_path=save_path)
