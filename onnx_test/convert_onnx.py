import sys, os
import argparse
import torch

from data_loader import datasets
from models.fast_scnn import get_fast_scnn, FastSCNN


def GetArgs():
    parser = argparse.ArgumentParser(description="",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weights", type=str, default="/media/xin/work/github_pro/seg_model/Fast-SCNN-pytorch/train_weights/s0.0.4/fast_scnn_wire_best_model.pth",help="model path")
    parser.add_argument("--output", type=str, default="onnx_model/fast_scnn_wire_best_argmax_256x640_no_random.onnx",help="output model path")
    parser.add_argument('--dataset', type=str, default='wire',
                        help='dataset name (default: citys)')

    args = parser.parse_args()
    return args

def main():
    # H, W = 1024, 2048
    # H, W = 480, 640
    # H, W = 640, 640
    H, W = 256, 640
    args = GetArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastSCNN(datasets[args.dataset].NUM_CLASS,test=True).to(device)
    model.load_state_dict(torch.load(args.weights))
    model.eval()
    # adaptive_avg_pool2d
    onnx_input = torch.rand(1, 3, H, W)
    onnx_input = onnx_input.to(device)
    torch.onnx.export(model,
                      onnx_input,
                      args.output,
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'])


if __name__ == '__main__':
    main()
