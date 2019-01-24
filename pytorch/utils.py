# Mohammad Saad
# utils.py
# Various utilities
import skimage.io as io
import numpy as np
from PIL import Image
from numpy.core._multiarray_umath import ndarray


def center_crop(image, cropX, cropY):
    y, x, d = image.shape
    startX = x // 2 - (cropX // 2)
    startY = y // 2 - (cropY // 2)
    return image[startY:startY+cropY, startX:startX+cropX, :]


def down_size(image,size_x,size_y):
    # Img = np.asarray(image)
#    io.imsave("test2.jpg", img__ / 255.0)
  #  Img = image.astype('float32')
    io.imsave("before_resize.jpg", image)
    Img = Image.open("before_resize.jpg")
    Img = Img.resize((size_x,size_y), Image.ANTIALIAS)
    # io.imsave("after_resize.jpg", Img)
    Img = np.array(Img)#.astype('float32')
    # io.imshow(Img / 255.0)
    # io.imsave("test3.jpg", Img / 255.0)
    # io.show()
    image = Img
    return image


