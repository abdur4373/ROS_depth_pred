# Mohammad Saad
# utils.py
# Various utilities
from PIL import Image
# from numpy.core._multiarray_umath import ndarray
from skimage import img_as_float

def center_crop(image, cropX, cropY):
    y, x, d = image.shape
    startX = x - cropX
    startY = y - cropY
    return image[(startY//2):y-(startY//2), (startX//2):x-(startX//2), :]


def down_size(image,size_x,size_y):
    # Img = np.asarray(image)
    #   io.imsave("test2.jpg", img__ / 255.0)
  #  Img = image.astype('float32')
  #   io.imsave("before_resize.jpg", image)
  #   Img = Image.open("before_resize.jpg")
    Img = image.resize((size_x,size_y), Image.ANTIALIAS)
    # io.imsave("after_resize.jpg", Img)
    Img = img_as_float(Img)#.astype('float32')
    # io.imshow(Img / 255.0)
    # io.imsave("test3.jpg", Img / 255.0)
    # io.show()

    return Img


