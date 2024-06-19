import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def process_image(image_path):
    # 打开图像
    img = Image.open(image_path)
    img_array = np.array(img)

    # 将像素值不为0的像素值改为7
    img_array[img_array != 0] = 7

    # 将处理后的数组转换回图像
    processed_img = Image.fromarray(img_array)

    # 保存处理后的图像，覆盖原图像
    processed_img.save(image_path)


def process_images_in_directory(input_dir):
    # 遍历输入目录及其子目录
    for root, _, files in os.walk(input_dir):
        for file in tqdm(files, desc="Processing images"):
            if file.endswith('.png'):
                # 构建完整的文件路径
                input_path = os.path.join(root, file)

                # 处理并保存图像
                process_image(input_path)
                # print(f"Processed and saved: {input_path}")


if __name__ == '__main__':
    # 设置输入目录
    input_directory = '/media/xin/data/data/seg_data/ours/train_data/gtFine/train/mask'  # 替换为实际的输入目录路径

    # 处理图像
    process_images_in_directory(input_directory)

