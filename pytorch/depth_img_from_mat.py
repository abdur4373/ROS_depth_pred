import h5py
import numpy as np


def depth_gt(test_image):
    path_to_depth = '/home/maq/PycharmProjects/Deeper-Depth-Prediction_2/pytorch/test.mat'
    f = h5py.File(path_to_depth)
    img = f['rgb_undist'][test_image]
    img_ = np.empty([480, 640, 3])
    img_[:, :, 0] = img[0, :, :].T
    img_[:, :, 1] = img[1, :, :].T
    img_[:, :, 2] = img[2, :, :].T
    img__ = img_.astype('float32')
    # io.imsave("test_image.jpg", img__ / 255.0)
    # io.imshow(img__ / 255.0)
    # io.show()
    depth = f['DepthFilled'][test_image]

    depth = depth.T.astype('float32')
    # print(depth.dtype)

    depth_gt_diff = depth - np.amin(depth)
    depth_gt_norm = depth_gt_diff / np.amax(depth_gt_diff)
    # print('np.amax(depth_gt_diff)', np.amax(depth_gt_diff))

    depth_ = np.empty([427, 561, 3])
    depth_[:, :, 0] = depth_gt_norm[:, :]
    depth_[:, :, 1] = depth_gt_norm[:, :]
    depth_[:, :, 2] = depth_gt_norm[:, :]
    # io.imshow(depth_)
    # io.imsave("depth_gt.jpg", depth_)
    # io.show()

    rgb_time_sec = int(f['rgb_time_sec'][test_image])

    # print rgb_time_sec, int(rgb_time_sec)

    rgb_time_nsec = int(f['rgb_time_nsec'][test_image])

    # print rgb_time_nsec, int(rgb_time_nsec)

    return depth, img__, rgb_time_sec, rgb_time_nsec
