import h5py
import numpy as np
import skimage.io as io
from utils import *


def depth_gt(i):
    path_to_depth = '/home/maq/PycharmProjects/Deeper-Depth-Prediction_2/pytorch/nyu_depth_v2_labeled .mat'
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
    img__ = img_.astype('float32')
    io.imsave("test2.jpg", img__ / 255.0)
    # io.imshow(img__/255.0)
    # io.show()

    # Img = Image.open("test2.jpg")
    # Img = Img.resize([304,228], Image.ANTIALIAS)
    # Img = np.array(Img).astype('float32')
    # # Img = np.array(Image.fromarray(img__).resize((304, 228), Image.ANTIALIAS)).astype('float32')
    # io.imshow(Img/255.0)
    # io.imsave("test3.jpg",Img/255.0)
    # io.show()
    # read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64

    depth = f['depths'][i]

    sliced_depth_gt = depth[16:624, 12:468]
    sliced_depth_gt = sliced_depth_gt.T

    sliced_depth_gt_diff = sliced_depth_gt - np.amin(sliced_depth_gt)
    sliced_depth_gt_norm = sliced_depth_gt_diff / np.amax(sliced_depth_gt_diff)

    depth_ = np.empty([456, 608, 3])
    depth_[:, :, 0] = sliced_depth_gt_norm[:, :]
    depth_[:, :, 1] = sliced_depth_gt_norm[:, :]
    depth_[:, :, 2] = sliced_depth_gt_norm[:, :]

    depth_scale = np.amax(sliced_depth_gt_diff)
    print(depth_scale)
    io.imsave("sliced_depth_gt.jpg", depth_)
    return sliced_depth_gt
    # print(depth_[:,:,1])
    # print('2')
    # print(depth_[:,:,2])

    # io.imshow(depth_/4.0)
    # io.imsave("test2_depth_gt.jpg", depth_/4.0)
    # io.show()
    # depth_gt = Image.open("test2_depthgt.jpg")
    # depth_resize_img_gt = down_size(depth_gt, 168, 134)
    # depth_cropped_img_gt = center_crop(depth_resize_img_gt, 160, 128)
    # scipy.misc.toimage(depth_cropped_img_gt, cmin=0.0, cmax=1.0).save('depth_cropped_gt.jpg')
    # pytorch_img_gt = torch.from_numpy(depth_cropped_img_gt).permute(2, 0, 1).unsqueeze(0).float()
    # # save_image(pytorch_img_gt, "depth_cropped_gt_input.jpg")
    #
    # depth_tensor_gt = Variable(pytorch_img_gt)
    #
    #
    # # print('size')
    # # print(list(depth_tensor_gt.size()))
    #
    # # Img = np.array(Img).astype('float32')
    # # # Img = np.array(Image.fromarray(img__).resize((304, 228), Image.ANTIALIAS)).astype('float32')
    # # io.imshow(Img/255.0)
    # # io.imsave("test3.jpg",Img/255.0)
    # # io.show()
    # # print(depth_tensor_gt)
    #
    #
    # return depth_tensor_gt

# depth_gt(3)
