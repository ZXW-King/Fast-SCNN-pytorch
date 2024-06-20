import os
import argparse
import torch
from data_loader import datasets
from torchvision import transforms
from models.fast_scnn import get_fast_scnn, FastSCNN
from PIL import Image
from utils.visualize import get_color_pallete

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--model', type=str, default='fast_scnn',
                    help='model name (default: fast_scnn)')
parser.add_argument('--dataset', type=str, default='wire',
                    help='dataset name (default: citys)')
parser.add_argument('--weights', default='./train_weights/fast_scnn_wire_best_model.pth',
                    help='Directory for saving checkpoint models')
parser.add_argument('--input-pic', type=str,
                    default='./dataset/02_6178896901985513.jpg',
                    help='path to the input picture')
parser.add_argument('--outdir', default='./test_result', type=str,
                    help='path to save the predict result')

parser.add_argument('--cpu', dest='cpu', action='store_true')
parser.set_defaults(cpu=False)

args = parser.parse_args()


def demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # image transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(args.input_pic).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    model = FastSCNN(datasets[args.dataset].NUM_CLASS).to(device)
    model.load_state_dict(torch.load(args.weights))
    print('Finished loading model!')
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        print(outputs[0].shape)
    pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
    mask = get_color_pallete(pred, args.dataset)
    outname = os.path.splitext(os.path.split(args.input_pic)[-1])[0] + '.png'
    mask.save(os.path.join(args.outdir, outname))


if __name__ == '__main__':
    demo()
