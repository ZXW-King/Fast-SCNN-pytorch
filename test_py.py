import cv2
import numpy as np
import torch
from PIL import Image
def _class_to_index(mask):
    key = np.array([1, 1, 1, 1, 1, 1,
                              1, 1, 0, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1])
    mapping = np.array(range(-1, len(key) - 1)).astype('int32')

    values = np.unique(mask)
    for value in values:
        assert (value in mapping)
    index = np.digitize(mask.ravel(), mapping, right=True)
    return key[index].reshape(mask.shape)

def get_crop_resize_img(img_path):
    # 使用 PIL 打开图像
    img = Image.open(img_path)
    # 打印原始图像的尺寸
    print(f"Original image size: {img.size}")  # 输出原始图像的尺寸 (width, height)
    print(img.size[1] - int(img.size[1] / 2 // 32 * 32 * 2))
    # 裁剪图像，去掉上方20行
    crop_box = (0, img.size[1] - int(img.size[1] / 2 // 32 * 32 * 2), img.size[0], img.size[1])
    cropped_img = img.crop(crop_box)

    # 打印裁剪后图像的尺寸
    print(f"Cropped image size: {cropped_img.size}")

    # 计算新尺寸，裁剪后的尺寸除以32并向下取整
    new_size = (cropped_img.size[0] // 32 * 16  , cropped_img.size[1] // 32 *16)

    # 调整图像大小
    resized_img = cropped_img.resize(new_size, Image.LANCZOS)

    # 打印调整大小后的图像尺寸
    print(f"Resized image size: {resized_img.size}")


def get_img_point():
    # 读取图像
    image = cv2.imread('/media/xin/work/github_pro/seg_model/Fast-SCNN-pytorch/dataset/02_6178896901985513.jpg',0)
    thresh1,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
    # 获取图像的高度和宽度
    height, width = image.shape

    # 获取所有像素点的坐标并存储在一个列表中
    for x in range(height):
        for y in range(width):
            print(type(image[x, y]))
            if image[x, y] == 0:
                print(image[x,y])

def is_img_same(img1,img2):
    # 读取图像
    image1 = cv2.imread(img1)
    image2 = cv2.imread(img2)

    # 比较图像是否大小相同
    if image1.shape == image2.shape:
        # 逐像素比较
        difference = cv2.subtract(image1, image2)
        b, g, r = cv2.split(difference)

        if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
            print("两张图像相同")
        else:
            print("两张图像不同")
    else:
        print("两张图像大小不同，无法比较")

if __name__ == '__main__':
    # mask = np.random.randint(7,22,(3,3))
    # # mask = np.full(3,0)
    # print(mask)
    # print(_class_to_index(mask))
    # get_crop_resize_img("/media/xin/work/github_pro/seg_model/Fast-SCNN-pytorch/dataset/02_6178896901985513.jpg")
    # get_img_point()
    is_img_same("dataset/same_img/947176970284.jpg", "dataset/same_img/947176970460.jpg")
