import numpy as np
from PIL import Image
import os
import re

size = (200,200)
threshold = 60


current_path = os.getcwd()
current_path = os.getcwd()
test_paths = []
path = current_path+"/train_pics/"
for i in os.listdir(path):
    if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
        test_paths.append(path+i)

count = 0
for a_file in test_paths:
    image = Image.open(a_file).convert(mode="L")  # grey scale (mode “L”)
    image = image.resize(size)
    imgArray = np.asarray(image,dtype=np.uint8)  # grey scale: 0~255 (8 bits)
    x = np.zeros(imgArray.shape,dtype=np.float)
    # grey scale to black & white
    x[imgArray > threshold] = 1
    x[imgArray <= threshold] = -1
    
    y = np.zeros(x.shape,dtype=np.uint8)
    y[x==1] = 255  # white
    y[x==-1] = 0  # black
    img = Image.fromarray(y,mode="L")
    global current_path
    outfile = current_path+"/train_"+str(count)+".jpeg"
    img.save(outfile)
    count += 1
  
