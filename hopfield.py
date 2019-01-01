import numpy as np
import random
from PIL import Image
import os
import re

class HopfieldNetwork:
    def __init__(self, sync, train_files, test_files, theta=0.5, time=1000, \
                 size=(100,100), threshold=60, current_path=None):

        self.w = np.zeros([size[0]*size[0],size[1]*size[1]])
        self.sync = sync
        self.train_files = train_files
        self.test_files = test_files
        self.theta = theta
        self.time = time
        self.size = size
        self.threshold = threshold
        self.current_path = current_path
        
        # create the weight matrix
        self.create_W()

        #Import test data
        self.test()


    def create_W(self):
        print('Importing train images and building the weight matrix...\n')
        inputs_matrix = np.zeros([len(self.train_files),self.size[0]*self.size[1]])
        i = 0
        for train_file in self.train_files:
            print(train_file)
            x = self.readImg2array(file=train_file, size=self.size, threshold=self.threshold)
            x_one_dimension = x.flatten()  # 2d numpy array to 1d numpy array (ready for input)
            print('Number of nodes = ', len(x_one_dimension))
            inputs_matrix[i] = x_one_dimension
            i += 1
        self.w = np.dot(np.transpose(inputs_matrix), inputs_matrix)/len(self.train_files)   
        print('Done.')


    def test(self):
        print('Importing test images and updating...')
        count = 0
        for path in self.test_files:
            y = self.readImg2array(file=path,size=self.size,threshold=self.threshold)
            two_dimension_shape = y.shape

            y_one_dimension = y.flatten()
      
            y_after = self.update_sync(y_one_dimension) if self.sync else self.update_async(y_one_dimension)
            print('Done.')
            y_after = y_after.reshape(two_dimension_shape)
            if self.current_path is not None:
                outfile = self.current_path+"/after_"+str(count)+".jpeg"
                after_img = self.array2img(y_after,outFile=outfile)
                print(outfile + ' saved.')
                after_img.show()
            count += 1
        print('All done.')


    # image to numpy array (white is 1, black is -1)
    def readImg2array(self, file, size, threshold= 145):
        image = Image.open(file).convert(mode="L")  # grey scale (mode “L”)
        image = image.resize(size)
        imgArray = np.asarray(image,dtype=np.uint8)  # grey scale: 0~255 (8 bits)
        x = np.zeros(imgArray.shape,dtype=np.float)
        # grey scale to black & white
        x[imgArray > threshold] = 1
        x[imgArray <= threshold] = -1
        return x


    # asynchronous
    def update_async(self,y):
        for s in range(self.time):
            m = len(y)
            i = random.randint(0,m-1)
            u = np.dot(self.w[i][:],y) - self.theta

            y[i] = 1 if u>0 else -1
        
        return y


    # synchronous
    def update_sync(self,y):
        for s in range(self.time):  
            output = np.dot(y, self.w) - self.theta
            y = np.array([1 if pixel>0 else -1 for pixel in output])

        return y


    # turn numpy array back to grey scale image
    def array2img(self, data, outFile = None):

        y = np.zeros(data.shape,dtype=np.uint8)
        y[data==1] = 255  # white
        y[data==-1] = 0  # black
        img = Image.fromarray(y,mode="L")
        if outFile is not None:
            img.save(outFile)
        return img


if __name__ == '__main__':
    # list of input file path
    current_path = os.getcwd()
    train_paths = []
    path = current_path+"/train_pics/"
    for i in os.listdir(path):
        if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
            train_paths.append(path+i)

    # list of test file path
    test_paths = []
    path = current_path+"/test_pics/"
    for i in os.listdir(path):
        if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
            test_paths.append(path+i)

    # Hopfield network 
    # choose synchronous or asynchronous
    h = HopfieldNetwork(sync=True, train_files=train_paths, test_files=test_paths, theta=0.5,time=1,\
             size=(200,200),threshold=60, current_path = current_path)