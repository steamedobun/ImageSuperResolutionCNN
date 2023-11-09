from keras.utils import load_img
from keras.utils import img_to_array
import numpy as np
from keras.models import load_model
import PIL
from PIL import Image


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

"""
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, None, None,  0           []                               
                                 1)]                                                                                                                                                              
 conv2d (Conv2D)                (None, None, None,   640         ['input_1[0][0]']                
                                64)                                                                                                                                                               
 conv2d_1 (Conv2D)              (None, None, None,   36928       ['conv2d[0][0]']                 
                                64)                                                                                                                                                               
 conv2d_2 (Conv2D)              (None, None, None,   36928       ['conv2d_1[0][0]']               
                                64)                                                                                                                                                               
 conv2d_3 (Conv2D)              (None, None, None,   36928       ['conv2d_2[0][0]']               
                                64)                                                                                                                                                               
 add (Add)                      (None, None, None,   0           ['conv2d_3[0][0]',               
                                64)                               'conv2d_1[0][0]']                                                                                                              
 conv2d_4 (Conv2D)              (None, None, None,   36928       ['add[0][0]']                    
                                64)                                                                                                                                                                
 conv2d_5 (Conv2D)              (None, None, None,   36928       ['conv2d_4[0][0]']               
                                64)                                                                                                                                                               
 add_1 (Add)                    (None, None, None,   0           ['conv2d_5[0][0]',               
                                64)                               'add[0][0]']                                                                                                                    
 conv2d_6 (Conv2D)              (None, None, None,   36928       ['add_1[0][0]']                  
                                64)                                                                                                                                                                
 conv2d_7 (Conv2D)              (None, None, None,   36928       ['conv2d_6[0][0]']               
                                64)                                                                                                                                                                 
 add_2 (Add)                    (None, None, None,   0           ['conv2d_7[0][0]',               
                                64)                               'add_1[0][0]']                                                                                                                    
 conv2d_8 (Conv2D)              (None, None, None,   36928       ['add_2[0][0]']                  
                                64)                                                                                                                                                                
 conv2d_9 (Conv2D)              (None, None, None,   5193        ['conv2d_8[0][0]']               
                                9)                                                                                                                                                                  
 tf.nn.depth_to_space (TFOpLamb  (None, None, None,   0          ['conv2d_9[0][0]']               
 da)                            1)                                                                                                                                                             
==================================================================================================
Total params: 301,257
Trainable params: 301,257
Non-trainable params: 0
__________________________________________________________________________________________________
"""

if __name__ == '__main__':

    # 加载模型
    model = load_model('./model/')

    # 加载图像
    img = load_img('./image_new.jpg')

    img_predict = upscale_image(model, img)

    # img_predict_2 = upscale_image(model, img_predict)

    # 打印预测结果
    img.show()
    img_predict.show()
    img_predict.save("./image_new_2.jpg")
    # img_predict_2.show()
    # img_predict_2.save()
