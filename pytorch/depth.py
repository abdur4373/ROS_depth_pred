import skimage.io as io
import numpy as np
import h5py
from utils import *
from torchvision.utils import save_image
from PIL import Image
import scipy.misc
import torch
from torch.autograd import Variable


def depth_gt(i):
    path_to_depth = '/media/ghost/8c5dd22b-3c0c-41d2-9807-c4094164ca3e/ghost/down/nyu_depth_v2_labeled .mat'
# read mat file
    f = h5py.File(path_to_depth)
    data = f.get('images')
    # read 0-th image. original format is [3 x 640 x 480], uint8
    img = f['images'][i]
    # # reshape
    img_ = np.empty([480, 640, 3])
    img_[:, :, 0] = img[0, :, :].T
    img_[:, :, 1] = img[1, :, :].T
    img_[:, :, 2] = img[2, :, :].T
    # imshow
    #
    # print(img_.dtype)
    # img="image1.jpg"
    # down_size(img,304,228)



    img__ = img_.astype('float32')
    # print(img__.dtype)

    # io.imshow(img__/255.0)
    # io.show()


    # io.imsave("test2.jpg",img__/255.0)
    # Img = Image.open("test2.jpg")
    # Img = Img.resize([304,228], Image.ANTIALIAS)
    # Img = np.array(Img).astype('float32')
    # # Img = np.array(Image.fromarray(img__).resize((304, 228), Image.ANTIALIAS)).astype('float32')
    # io.imshow(Img/255.0)
    # io.imsave("test3.jpg",Img/255.0)
    # io.show()
    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
    depth = f['depths'][i]
    # reshape for imshow
    depth_ = np.empty([480, 640, 3])
    depth_[:, :, 0] = depth[:, :].T
    depth_[:, :, 1] = depth[:, :].T
    depth_[:, :, 2] = depth[:, :].T
    print('size')
    print(depth)
    print('depth')
    print(depth.dtype)
    print('1')
    # print(depth_[:,:,1])
    # print('2')
    # print(depth_[:,:,2])

    # io.imshow(depth_/4.0)
    # io.imsave("test2_depth_gt.jpg", depth_/4.0)
    # io.show()
    depth_gt = Image.open("test2_depthgt.jpg")
    depth_resize_img_gt = down_size(depth_gt, 168, 134)
    depth_cropped_img_gt = center_crop(depth_resize_img_gt, 160, 128)
    scipy.misc.toimage(depth_cropped_img_gt, cmin=0.0, cmax=1.0).save('depth_cropped_gt.jpg')
    pytorch_img_gt = torch.from_numpy(depth_cropped_img_gt).permute(2, 0, 1).unsqueeze(0).float()
    # save_image(pytorch_img_gt, "depth_cropped_gt_input.jpg")

    depth_tensor_gt = Variable(pytorch_img_gt)


    # print('size')
    # print(list(depth_tensor_gt.size()))

    # Img = np.array(Img).astype('float32')
    # # Img = np.array(Image.fromarray(img__).resize((304, 228), Image.ANTIALIAS)).astype('float32')
    # io.imshow(Img/255.0)
    # io.imsave("test3.jpg",Img/255.0)
    # io.show()
    # print(depth_tensor_gt)
    return depth_tensor_gt


depth_gt(3)