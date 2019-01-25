import skimage.io as io
import numpy as np
import h5py
# from utils import*
from torchvision.utils import save_image
from PIL import Image

path_to_depth = '/media/ghost/8c5dd22b-3c0c-41d2-9807-c4094164ca3e/ghost/down/nyu_depth_v2_labeled .mat'
# read mat file
f = h5py.File(path_to_depth)
data = f.get('images')
# # read 0-th image. original format is [3 x 640 x 480], uint8
# # img = f['images'][1]
# # # reshape
# # img_ = np.empty([480, 640, 3])
# # img_[:,:,0] = img[0,:,:].T
# # img_[:,:,1] = img[1,:,:].T
# # img_[:,:,2] = img[2,:,:].T
# imshow
#
# print(img_.dtype)
# img="image1.jpg"
# down_size(img,304,228)



img__ = img_.astype('float32')
print(img__.dtype)
io.imshow(img__/255.0)
io.show()
# io.imsave("test2.jpg",img__/255.0)
# Img = Image.open("test2.jpg")
# Img = Img.resize([304,228], Image.ANTIALIAS)
# Img = np.array(Img).astype('float32')
# # Img = np.array(Image.fromarray(img__).resize((304, 228), Image.ANTIALIAS)).astype('float32')
# io.imshow(Img/255.0)
# io.imsave("test3.jpg",Img/255.0)
# io.show()
# read corresponding depth (aligned to the image, in-painted) of size [640 x 480], float64
# depth = f['depths'][1]
# # reshape for imshow
# depth_ = np.empty([480, 640, 3])
# depth_[:,:,0] = depth[:,:].T
# depth_[:,:,1] = depth[:,:].T
# depth_[:,:,2] = depth[:,:].T
# io.imshow(depth_/4.0)
# io.imsave("test2_depthgt.jpg",depth_/4.0)
# io.show()