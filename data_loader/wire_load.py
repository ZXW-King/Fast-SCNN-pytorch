"""wire Dataloader"""
import os
import random

import cv2
import numpy as np
import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['WireSegmentation']


class WireSegmentation(data.Dataset):
    """Cityscapes Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to Cityscapes folder. Default is './datasets/citys'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    >>> ])
    >>> # Create Dataset
    >>> trainset = WireSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    BASE_DIR = 'wire'
    NUM_CLASS = 3

    def __init__(self, root='./datasets/citys', split='train', mode=None, transform=None,
                 base_size=256, crop_size=640, version='s0.0.1',**kwargs):
        super(WireSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.mode = mode if mode is not None else split
        self.transform = transform
        self.base_size = base_size
        self.crop_size = crop_size
        self.version = version
        self.images, self.mask_paths = _get_city_pairs(self.root, self.version,self.split)
        assert (len(self.images) == len(self.mask_paths))
        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: " + self.root + "\n")
        # self._key = np.array([0, 0, 0, 0, 0, 0,
        #                       0, 0, 1, 0, 0, 0,
        #                       0, 0, 0, 0, 0, 0,
        #                       0, 0, 0, 0, 0, 0,
        #                       0, 0, 0, 0, 0, 0,
        #                       0, 0, 0, 0, 0])
        self._key = np.array([1, 1, 1, 1, 1, 1,
                              1, 1, 0, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')

    def _class_to_index(self, mask):
        values = np.unique(mask)
        for value in values:
            assert (value in self._mapping)
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        return self._key[index].reshape(mask.shape)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.get_crop_img(img)
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.mask_paths[index])
        mask = self.get_crop_img(mask)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask

    def get_crop_resize_img(self,img):
        # 裁剪图像，去掉上方20行
        crop_box = (0, img.size[1] - int(img.size[1] / 2 // 32 * 32 * 2), img.size[0], img.size[1])
        cropped_img = img.crop(crop_box)
        # 计算新尺寸，裁剪后的尺寸除以32并向下取整
        new_size = (cropped_img.size[0] // 32 * 16, cropped_img.size[1] // 32 * 16)
        # 调整图像大小
        resized_img = cropped_img.resize(new_size, Image.LANCZOS)
        return resized_img

    def get_crop_img(self,img):
        # 裁剪图像
        crop_box = (0, 224, img.size[0], img.size[1])
        cropped_img = img.crop(crop_box)
        return cropped_img

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1 + outsize, y1 + outsize))
        mask = mask.crop((x1, y1, x1 + outsize, y1 + outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror  随机左右镜像翻转
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge)  # 随机缩放
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop  # 随机裁剪
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0


def _get_city_pairs(folder, version,split='train'):
    def get_path_pairs(txt_files):
        img_paths = []
        mask_paths = []
        with open(txt_files) as f:
            list_f = f.readlines()
            for img_path in list_f:
                imgpath = img_path.strip()
                if imgpath.endswith(".png") or imgpath.endswith(".jpg"):
                    maskname = imgpath.replace('leftImg8bit', 'gtFine')
                    maskpath = maskname.replace(".jpg", ".png")
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    # else:
                        # print('cannot find the mask or image:', imgpath, maskpath)

        print('Found {} type {} images in the folder'.format(split,len(img_paths)))
        return img_paths, mask_paths

    if split in ('train', 'val'):
        txt_files = os.path.join(folder,'Record_data/'+ version+"/"+split+".txt")
        img_paths, mask_paths = get_path_pairs(txt_files)
        return img_paths, mask_paths
    else:
        assert split == 'trainval'
        print('trainval set')
        return


if __name__ == '__main__':
    dataset = WireSegmentation(root="/media/xin/data/data/seg_data/ours/train_data",split="train")
    paused = False
    for i in range(len(dataset)):
        if not paused:
            img, label = dataset[i]
            # print(img)
            label = label * 255
            label = np.array(label).astype(np.uint8)
            print(img.shape,label.shape)
            # t2, label = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY_INV)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.uint8)
            # 拼接原始图像和处理后的label图像
            # combined_img = np.hstack((img, label))
            # 使用OpenCV显示拼接后的图像
            # cv2.imshow('Combined Image', combined_img)
            cv2.imshow("image",img)
            cv2.imshow('Combined Image', label)

        key = cv2.waitKey()
        if key == 27:  # 按下Esc键暂停
            paused = not paused
        elif key == 32:  # 按下空格键继续
            paused = not paused
        elif key == ord('q'):  # 按下q键退出
            break
    cv2.destroyAllWindows()