import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F


class CustomCompose(transforms.Compose):
    def __init__(self, transforms):
        super().__init__(transforms=transforms)

    def __call__(self, img, **kwargs):
        for t in self.transforms:
            img = t(img, kwargs)
        return img

    def __repr__(self):
        return super().__repr__()


class CustomNormalize(transforms.Normalize):
    def __init__(self, mean, std, inplace=False):
        super().__init__(mean=mean, std=std, inplace=inplace)

    def __call__(self, tensor, *args, **kwargs):
        return super().__call__(tensor)

    def __repr__(self):
        return super().__repr__()


class CustomRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def __call__(self, img, *args, **kwargs):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()


class CustomRandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def __call__(self, img, *args, **kwargs):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()


class CustomResize(transforms.Resize):
    def __init__(self, size, interpolation=F.InterpolationMode.BILINEAR):
        super().__init__(size=size, interpolation=interpolation)

    def __call__(self, img, *args, **kwargs):
        return super().__call__(img)

    def __repr__(self):
        return super().__repr__()


class ClearBorders(object):
    def __init__(self, size):
        self.size = size
        self.cropper = transforms.CenterCrop(self.size)

    def __call__(self, img):
        img = self.cropper(img)
        fill = int(np.percentile(np.array(img), 1))
        img = transforms.Pad(self.size // 2, fill=fill)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class CustomCenteredCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        fill = int(np.percentile(np.array(img), 1))
        img = transforms.Pad(self.size, fill=fill)(img)
        img = transforms.CenterCrop(self.size)(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class RandomGamma(object):
    def __init__(self, limit=[0.5, 1.5]):
        self.limit = limit

    def __call__(self, img):
        gamma = np.random.uniform(self.limit[0], self.limit[1])
        return F.adjust_gamma(img=img, gamma=gamma)

    def __repr__(self):
        return self.__class__.__name__ + "(gamma={})".format(self.limit)


class CustomRandomRotation(object):
    def __init__(self, degrees, fill=None):
        self.degrees = degrees
        self.fill = fill

    def __call__(self, img):
        if self.fill is None:
            fill = int(np.percentile(np.array(img), 1))
        else:
            fill = self.fill
        return transforms.RandomRotation(
            self.degrees, fill=fill, interpolation=F.InterpolationMode.BICUBIC
        )(img)

    def __repr__(self):
        return self.__class__.__name__ + "(degrees={}, fill={})".format(
            self.degrees, self.fill
        )


class CustomPad(object):
    def __init__(self, size, fill=None):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        w, h = np.array(img).shape
        if self.fill is None:
            fill = int(np.percentile(np.array(img), 1))
        else:
            fill = self.fill
        return F.pad(
            img, padding=[(self.size - h) // 2 + 1, (self.size - h) // 2 + 1], fill=fill
        )

    def __repr__(self):
        return self.__class__.__name__ + "(size={})".format(self.size)


class ToRGBTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, img, *args, **kwargs):
        return transforms.ToTensor()(img).repeat(3, 1, 1)
