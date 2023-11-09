import cv2
from PIL import Image
from predict import upscale_image
from keras.models import load_model
import numpy as np

# 视频读取及输出路径
video_path = "./video_main/input_video.mp4"
output_path = "./video_main/output_video.mp4"

# 获取视频帧率及尺寸
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# 加载模型
model = load_model('./model/')

# 逐帧处理视频
for i in range(total_frames):
    ret, frame = cap.read()
    if not ret:
        break

    # 转换图像为统一尺寸
    resized = cv2.resize(frame, (640, 480))

    # 将图像转换为PIL图像，保存为png文件
    image = Image.fromarray(resized[..., ::-1])

    processed_image = upscale_image(model, image)

    # 将数据类型转换为numpy数组
    processed_image = np.array(processed_image, dtype=np.uint8)

    # 将numpy数组转换为BGR格式图像
    bgr_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)

    # 写入视频
    out.write(bgr_image)

    # 释放资源
    cv2.waitKey(1)

# 释放资源
cap.release()
out.release()
