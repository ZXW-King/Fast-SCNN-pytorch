import argparse

import cv2
import torch

from onnx_test.onnxmodel import ONNXModel
from utils.visualize import get_color_pallete
import numpy as np



def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--image", type=str, default="../dataset/ulm_000003_000019_leftImg8bit.png",help="")
    parser.add_argument("--model", type=str, default="fast_scnn.onnx",help="")
    parser.add_argument('--dataset', type=str, default='citys',
                        help='dataset name (default: citys)')

    args = parser.parse_args()
    return args


def pad_or_crop_to_target_size(image, target_height, target_width):
    h, w = image.shape[:2]

    if h > target_height or w > target_width:
        # 中心裁剪
        top = (h - target_height) // 2
        bottom = top + target_height
        left = (w - target_width) // 2
        right = left + target_width
        cropped_image = image[top:bottom, left:right]
        return cropped_image, top, bottom, left, right, True
    else:
        # 填充
        top = max((target_height - h) // 2, 0)
        bottom = max(target_height - h - top, 0)
        left = max((target_width - w) // 2, 0)
        right = max(target_width - w - left, 0)
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded_image, top, bottom, left, right, False


def unpad_or_uncrop_image(image, top, bottom, left, right, is_cropped, original_height, original_width):
    if is_cropped:
        # 因为是裁剪操作，直接返回原始大小的图像
        return cv2.resize(image, (original_width, original_height), cv2.INTER_LANCZOS4)
    else:
        # 去除填充
        return image[top:image.shape[0] - bottom, left:image.shape[1] - right]


def preprocess_image(image_path, W, H):
    img_org = cv2.imread(image_path)
    org_H, org_W = img_org.shape[:2]
    img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)

    # Padding or cropping the image to target size
    img_processed, top, bottom, left, right, is_cropped = pad_or_crop_to_target_size(img, H, W)

    # Convert to float32 and normalize
    img_processed = img_processed.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_processed = (img_processed - mean) / std

    # Convert to NCHW format
    img_processed = img_processed.transpose(2, 0, 1)
    img_processed = np.expand_dims(img_processed, axis=0).astype("float32")

    return img_processed, org_H, org_W, top, bottom, left, right, is_cropped


def test_onnx(img_path, model_file, W, H, dataset):
    model = ONNXModel(model_file)
    img_processed, org_H, org_W, top, bottom, left, right, is_cropped = preprocess_image(img_path, W, H)

    output = model.forward(img_processed)
    out = output[0][0]  # (C, H, W)
    out = out.transpose(1, 2, 0)  # (H, W, C)

    # Unpad or uncrop the output to original size
    out_unprocessed = unpad_or_uncrop_image(out, top, bottom, left, right, is_cropped, org_H, org_W)

    # If uncropped, resize to original size
    if not is_cropped:
        out_resized = cv2.resize(out_unprocessed, (org_W, org_H), cv2.INTER_LANCZOS4)
    else:
        out_resized = out_unprocessed

    out_resized = out_resized.transpose(2, 0, 1)  # (C, H, W)
    res = np.expand_dims(out_resized, axis=0)  # (N, C, H, W)

    print(res.shape)
    pred = torch.argmax(torch.from_numpy(res), 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, dataset)
    mask.save("res1.png")

def main():
    args = GetArgs()
    test_onnx(args.image, args.model,640,480,args.dataset)

if __name__ == '__main__':
    main()