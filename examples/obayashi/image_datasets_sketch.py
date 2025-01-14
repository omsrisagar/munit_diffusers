import math
import random

from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms as transforms
import torch as th
from degradation.bsrgan_light import degradation_bsrgan_variant as degradation_fn_bsr_light
from functools import partial
import cv2
import json

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)

def load_data_sketch(
    *,
    # data_dir,
    all_files,
    text_dict,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=True,
    train=True,
    low_res = 0,
    uncond_p = 0,
    mode = ''
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    # if not data_dir:
    #     raise ValueError("unspecified data directory")
    # with open(data_dir) as f:
    #     all_files = f.read().splitlines()
    # all_files = _list_image_files_recursively(data_dir)

    print(len(all_files))
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    dataset = ImageDataset(
        image_size,
        all_files,
        text_dict,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_crop=random_crop,
        random_flip=train,
        down_sample_img_size=low_res,
        uncond_p=uncond_p,
        mode=mode,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True, pin_memory=False
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=False
        )
    while True:
        yield from loader
    # return loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        text_dict,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
        down_sample_img_size = 0,
        uncond_p = 0,
        mode = '',
    ):
        super().__init__()
        self.crop_size = 256 # why are these both hard coded instead of setting to resolution?
        self.resize_size = 256
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip
        self.text_dict = text_dict
      
        self.down_sample_img = partial(degradation_fn_bsr_light, sf=resolution//down_sample_img_size) if down_sample_img_size else None
        self.uncond_p = uncond_p
        self.mode = mode
        self.resolution = resolution

    def __len__(self):
        return  len(self.local_images)

    def __getitem__(self, idx):
        if self.mode == 'edge':
            # path = self.local_images[idx].replace('COCO-STUFF', 'COCO-Sketch')[:-4] + '.png'
            path = self.local_images[idx]
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            dirname2 = dirname.replace('_img', '_sketch', 1)
            path2 = os.path.join(dirname2, basename)
            assert os.path.exists(path2)
        # elif self.mode == 'flickr-edge':
        #     path = self.local_images[idx].replace('images', 'img256')[:-4] + '.png'
        #     path2 = path.replace('img256', 'sketch256')


        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")


        with bf.BlobFile(path2, "rb") as f:
            pil_image2 = Image.open(f)
            pil_image2.load()
            pil_image2 = pil_image2.convert("L")

        # transforms to resize and crop and filp to transform pil_image2 to resize_size and crop_size. 256x256
        params =  get_params(pil_image2.size, self.resize_size, self.crop_size)
        transform_label = get_transform(params, self.resize_size, self.crop_size, method=Image.NEAREST, crop =self.random_crop, flip=self.random_flip)
        label_pil = transform_label(pil_image2)

        im_dist = cv2.distanceTransform(255-np.array(label_pil), cv2.DIST_L1, 3)
        im_dist = np.clip((im_dist) , 0, 255).astype(np.uint8)
        im_dist = Image.fromarray(im_dist).convert("RGB")

        label_tensor =  get_tensor()(im_dist)[:1]
        label_tensor_ori =  get_tensor()(label_pil.convert('RGB')) 
 
        transform_image = get_transform( params, self.resize_size, self.crop_size, crop =self.random_crop, flip=self.random_flip)
        image_pil = transform_image(pil_image)
        if self.resolution < 256:
            image_pil = image_pil.resize((self.resolution, self.resolution), Image.BICUBIC) # why no resizing for label?
        image_tensor = get_tensor()(image_pil)

        if self.down_sample_img:
            image_pil = np.array(image_pil).astype(np.uint8)
            down_sampled_image = self.down_sample_img(image=image_pil)["image"]
            down_sampled_image = get_tensor()(down_sampled_image)
            data_dict = {"ref":label_tensor, "low_res":down_sampled_image, "ref_ori":label_tensor_ori, "path": path}
            data_dict.update({'text': self.text_dict[os.path.basename(path)] if self.text_dict else ''})
            return image_tensor, data_dict

        if random.random() < self.uncond_p: # unconditional (null condition) training
            label_tensor = th.ones_like(label_tensor)
        data_dict = {"ref":label_tensor, "ref_ori":label_tensor_ori, "path": path}
        data_dict.update({'text': self.text_dict[os.path.basename(path)]  if self.text_dict else ''})

        return image_tensor, data_dict

def get_params( size,  resize_size,  crop_size):
    w, h = size
    new_h = h
    new_w = w

    ss, ls = min(w, h), max(w, h)  # shortside and longside
    width_is_shorter = w == ss
    ls = int(resize_size * ls / ss)
    ss = resize_size
    new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - crop_size))
    y = random.randint(0, np.maximum(0, new_h - crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}
 

def get_transform(params,  resize_size,  crop_size, method=Image.BICUBIC,  flip=True, crop = True):
    transform_list = []
 
    transform_list.append(transforms.Lambda(lambda img: __scale(img, crop_size, method)))

    if flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
 
    return transforms.Compose(transform_list)

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def __scale(img, target_width, method=Image.BICUBIC):
    return img.resize((target_width, target_width), method)

def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img