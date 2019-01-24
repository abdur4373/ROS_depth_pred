import numpy as np
import skimage.io as io

from PIL import Image


Img = Image.open("test2.jpg")
Img = Img.resize([304,228], Image.ANTIALIAS)
Img = np.array(Img).astype('float32')



io.imshow(Img/255.0)
io.imsave("test3.jpg",Img/255.0)
io.show()
