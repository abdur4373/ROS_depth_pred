# Mohammad Saad
# utils.py
# Various utilities
from PIL import Image
# from numpy.core._multiarray_umath import ndarray
from skimage import img_as_float
import skimage.io as io
from scipy import interpolate
import numpy as np



def center_crop(image, cropX, cropY):
    y, x, d = image.shape
    startX = x - cropX
    startY = y - cropY
    return image[(startY//2):y-(startY//2), (startX//2):x-(startX//2), :]


def resize_size(image, size_x, size_y):
    # Img = np.asarray(image)
    # io.imsave("test2.jpg", img__ / 255.0)
    # Img = image.astype('float32')
    io.imsave("before_resize.jpg", image / 255.0)
    Img = Image.open("before_resize.jpg")
    resize_img = Img.resize((size_x, size_y), Image.ANTIALIAS)
    # io.imsave("after_resize.jpg", Img)
    resize_img = img_as_float(resize_img)  # .astype('float32')
    # io.imshow(Img / 255.0)
    # io.imsave("test3.jpg", Img / 255.0)
    # io.show()

    return resize_img


def resize_depth_pred(out_img_pred_np):
    x = [x for x in range(160)]
    y = [y for y in range(128)]

    f = interpolate.interp2d(x, y, out_img_pred_np)

    # xnew = np.linspace(0, 160, 608)
    xnew = np.linspace(0, 160, 561)

    # ynew = np.linspace(0, 128, 456)
    ynew = np.linspace(0, 128, 427)
    depth_pred_inter = f(xnew, ynew)

    # print("Predicted depth values of size 608,456 {0}".format(depth_pred_inter))
    # print('shape')
    # print(depth_pred_inter.shape)
    depth_pred_inter_diff = depth_pred_inter - np.amin(depth_pred_inter)
    depth_pred_inter_norm = depth_pred_inter_diff / np.amax(depth_pred_inter_diff)
    # print('ndepth_pred_inter_diff', np.amax(depth_pred_inter_diff))
    # depth_pred_inter_ = np.empty([456, 608, 3])
    depth_pred_inter_ = np.empty([427, 561, 3])
    depth_pred_inter_[:, :, 0] = depth_pred_inter_norm[:, :]
    depth_pred_inter_[:, :, 1] = depth_pred_inter_norm[:, :]
    depth_pred_inter_[:, :, 2] = depth_pred_inter_norm[:, :]
    # io.imshow(depth_pred_inter_ / 4.0)
    # io.imsave('depth_pred_inter.jpg', depth_pred_inter_)

    # io.show()

    return depth_pred_inter


def resize_depth_gt(depth_gt_out):
    x = [x for x in range(561)]
    y = [y for y in range(427)]

    f = interpolate.interp2d(x, y, depth_gt_out)

    # xnew = np.linspace(0, 160, 608)
    xnew = np.linspace(0, 561, 640)

    # ynew = np.linspace(0, 128, 456)
    ynew = np.linspace(0, 427, 480)
    depth_gt_inter = f(xnew, ynew)

    # print("Predicted depth values of size 608,456 {0}".format(depth_pred_inter))
    # print('shape')
    # print(depth_pred_inter.shape)
    depth_gt_inter_diff = depth_gt_inter - np.amin(depth_gt_inter)
    depth_gt_inter_norm = depth_gt_inter_diff / np.amax(depth_gt_inter_diff)
    # print('ndepth_pred_inter_diff', np.amax(depth_pred_inter_diff))
    # depth_pred_inter_ = np.empty([456, 608, 3])
    depth_gt_inter_ = np.empty([480, 640, 3])
    depth_gt_inter_[:, :, 0] = depth_gt_inter_norm[:, :]
    depth_gt_inter_[:, :, 1] = depth_gt_inter_norm[:, :]
    depth_gt_inter_[:, :, 2] = depth_gt_inter_norm[:, :]
    # io.imshow(depth_pred_inter_ / 4.0)
    # io.imsave('depth_pred_inter.jpg', depth_pred_inter_)

    # io.show()

    return depth_gt_inter


def append_nan_depth_gt(depth_gt_out):
    depth_array = np.empty([480, 640])
    depth_array[:] = np.nan

    # depth_array[1:1 + 2, 2:2 + 2] = depth_gt_out
    depth_array[
    (depth_array.shape[0] - depth_gt_out.shape[0]) / 2:((depth_array.shape[0] - depth_gt_out.shape[0]) / 2) +
                                                       depth_gt_out.shape[0],
    (depth_array.shape[1] - depth_gt_out.shape[1]) / 2:((depth_array.shape[1] - depth_gt_out.shape[1]) / 2) +
                                                       depth_gt_out.shape[
                                                           1]] = depth_gt_out

    depth_gt = depth_array.astype('float32')

    return depth_gt
