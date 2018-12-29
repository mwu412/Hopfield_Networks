import os

image = Image.open(file).convert(mode="L")  # grey scale (mode “L”)
image = image.resize(size)
imgArray = np.asarray(image,dtype=np.uint8)  # grey scale: 0~255 (8 bits)
x = np.zeros(imgArray.shape,dtype=np.float)
# grey scale to black & white
x[imgArray > threshold] = 1
x[x == 0] = -1
return x