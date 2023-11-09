import os
import tensorflow as tf
import keras
from keras import layers
from keras.utils import load_img
from keras.utils import img_to_array

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
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
    plt.savefig(str(prefix) + "-" + title + ".png")
    plt.show()


if __name__ == '__main__':
    # 读取当前项目文件夹下的数据集
    root_dir = "./datasets/BSR/BSDS500/data"

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
    # 评估模型性能
    total_bicubic_psnr = 0.0
    total_test_psnr = 0.0

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

    checkpoint_filepath = "./model_checkpoint/"
    model.load_weights(checkpoint_filepath)

    # 测试集中选择前10张图片进行评估
    for index, test_img_path in enumerate(test_img_paths[50:60]):
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
