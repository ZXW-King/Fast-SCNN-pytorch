import sys, os

import argparse

import torch

from onnx_test.onnxmodel import ONNXModel
import cv2
import numpy as np

from utils.visualize import get_color_pallete

W, H = 2048, 1024


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, default="../dataset/ulm_000003_000019_leftImg8bit.png",help="")
    parser.add_argument("--model", type=str, default="fast_scnn.onnx",help="")
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (default: citys)')

    args = parser.parse_args()
    return args


def test_onnx(img_path, model_file,dataset):
    model = ONNXModel(model_file)
    img_org = cv2.imread(img_path)
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    # 将图像转换为 float32 类型并归一化到 [0, 1]
    image = img.astype(np.float32) / 255.0

    # 定义均值和标准差
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 归一化
    image -= mean
    image /= std
    img = image.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype("float32")

    output = model.forward(img)
    pred = torch.argmax(torch.from_numpy(output[0]), 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, dataset)
    mask.save("res.png")

def main():
    args = GetArgs()
    test_onnx(args.image, args.model,args.dataset)


if __name__ == '__main__':
    main()
