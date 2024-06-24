import sys, os

import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from onnx_test.onnxmodel import ONNXModel
import cv2
import numpy as np

from utils.visualize import get_color_pallete



def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--image", type=str, default="../dataset/02_6178896901985513.jpg",help="")
    parser.add_argument("--image", type=str, default="/media/xin/data/data/seg_data/ours/ORIGIN/20240617_wire/test_select.txt",help="")
    # parser.add_argument("--model", type=str, default="onnx_model/fast_scnn_wire_best.onnx",help="")
    parser.add_argument("--model", type=str, default="onnx_model/fast_scnn_wire_best_argmax_224x320.onnx",help="")
    parser.add_argument('--dataset', type=str, default='wire',help='dataset name (default: citys)')
    parser.add_argument('--show_rgb', action='store_true')
    args = parser.parse_args()
    return args


def img_resize(image,target_size):
    H,W,_ = image.shape
    t_h = target_size[0] * 2
    diff_h = H - t_h
    cropped_img = image[diff_h:H,:]
    resize_img = cv2.resize(cropped_img,(target_size[1],target_size[0]))
    return resize_img


def test_onnx(img_path, model_file,dataset,show_rgb=True):
    model = ONNXModel(model_file)
    img_org = cv2.imread(img_path)
    img_res = img_resize(img_org,(224,320))
    img = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
    # 将图像转换为 float32 类型并归一化到 [0, 1]
    image = img.astype(np.float32) / 255.0

    # 定义均值和标准差
    mean = [0.485, 0.456, 0.406] # 123.675  116.28  103.53
    std = [0.229, 0.224, 0.225] # 58.395  57.12  57.375

    # 归一化
    image -= mean
    image /= std
    img = image.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype("float32")

    output = model.forward(img)
    # arg = torch.argmax(torch.from_numpy(output[0]), 1)
    # pred = arg.squeeze(0).cpu().data.numpy()
    pred = output[0].astype(np.float32)
    pred = cv2.resize(pred,(320*2,224*2))
    pred = np.pad(pred, ((480-224*2, 0),(0,0)), mode='constant', constant_values=1)
    if not show_rgb:
        cv_image = (pred * 255).astype(np.uint8)
        img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
        cv2.imshow("img",img_org)
        return cv_image
    else:
        mask = get_color_pallete(pred, dataset)
        # 将调色板图像转换为 RGB
        out_img = mask.convert("RGB")
        # 将 PIL Image 转换为 NumPy 数组
        cv_image = np.array(out_img)
        # 将 RGB 转换为 BGR（因为 OpenCV 使用 BGR）
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # 水平方向合并两张图像
    merged_image = np.hstack((img_org, cv_image))
    return merged_image

def main():
    args = GetArgs()
    img_path = args.image
    show_rgb = args.show_rgb
    if img_path.endswith(".txt"):
        with open(img_path) as f:
            for img in f:
                merged_image = test_onnx(img.strip(), args.model,args.dataset,show_rgb=show_rgb)
                cv2.imshow('Converted Image', merged_image)
                cv2.waitKey(500)
    else:
        merged_image = test_onnx(img_path,args.model,args.dataset,show_rgb=show_rgb)
        cv2.imshow('Converted Image', merged_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
