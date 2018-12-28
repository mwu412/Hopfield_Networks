import numpy as np
import random
from PIL import Image
import os
import re

class HopfieldNetwork:
    def __init__(self, train_files, test_files, theta=0.5, time=1000, \
                 size=(100,100), threshold=60, current_path=None):
        print("Importing images and creating weight matrix....")
        self.w = np.zeros([size[0]*size[0],size[1]*size[1]])
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
        count = 0
        for path in test_files:
            y = self.readImg2array(file=path,size=size,threshold=threshold)
            oshape = y.shape
            # y_img = self.array2img(y)
            # y_img.show()
            # print("Imported test data")

            y_one_dimension = y.flatten()
            print("Updating...")
      
            y_after = self.update(y=y_one_dimension)
            y_after = y_after.reshape(oshape)
            if current_path is not None:
                outfile = current_path+"/after_"+str(count)+".jpeg"
                after_img = self.array2img(y_after,outFile=outfile)
                after_img.show()
            count += 1

    def create_W(self):
        inputs_matrix = np.zeros([len(self.train_files),self.size[0]*self.size[1]])
        i = 0
        for train_file in self.train_files:
            print(train_file)
            x = self.readImg2array(file=train_file, size=self.size, threshold=self.threshold)
            x_one_dimension = x.flatten()
            print('Number of nodes = ', len(x_one_dimension))
            inputs_matrix[i] = x_one_dimension
            i += 1
            # add weights up for each training in put
        ######## whether to be divided by len(train_files) is a good question
        self.w = np.dot(np.transpose(inputs_matrix), inputs_matrix)   
        print("Weight matrix is done!!")

    
    #convert matrix to a vector (one dimension array)
    def mat2vec(self, x):
        m = x.shape[0]*x.shape[1]
        tmp1 = np.zeros(m)

        c = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tmp1[c] = x[i,j]
                c +=1
        return tmp1


    #Read Image file and convert it to Numpy array
    def readImg2array(self, file,size, threshold= 145):
        pilIN = Image.open(file).convert(mode="L")
        pilIN= pilIN.resize(size)
        #pilIN.thumbnail(size,Image.ANTIALIAS)
        imgArray = np.asarray(pilIN,dtype=np.uint8)
        x = np.zeros(imgArray.shape,dtype=np.float)
        x[imgArray > threshold] = 1
        x[x==0] = -1
        return x

    #Convert Numpy array to Image file like Jpeg
    def array2img(self, data, outFile = None):

        #data is 1 or -1 matrix
        y = np.zeros(data.shape,dtype=np.uint8)
        y[data==1] = 255
        y[data==-1] = 0
        img = Image.fromarray(y,mode="L")
        if outFile is not None:
            img.save(outFile)
        return img


    # asynchronous
    def update_asyn(self,y):
        for s in range(self.time):
            m = len(y)
            i = random.randint(0,m-1)
            u = np.dot(self.w[i][:],y) - self.theta

            if u > 0:
                y[i] = 1
            elif u < 0:
                y[i] = -1

        return y


    # synchronous
    def update(self,y):
        for s in range(self.time):
        
            u = np.dot(y, self.w) - self.theta

            if u > 0:
                y[i] = 1
            elif u < 0:
                y[i] = -1

        return y


#Main
if __name__ == '__main__':
    #First, you can create a list of input file path
    current_path = os.getcwd()
    train_paths = []
    path = current_path+"/train_pics/"
    for i in os.listdir(path):
        if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
            train_paths.append(path+i)

    #Second, you can create a list of sungallses file path
    test_paths = []
    path = current_path+"/test_pics/"
    for i in os.listdir(path):
        if re.match(r'[0-9a-zA-Z-_]*.jp[e]*g',i):
            test_paths.append(path+i)

    #Hopfield network starts!
    h = HopfieldNetwork(train_files=train_paths, test_files=test_paths, theta=0.5,time=2000,\
             size=(200,200),threshold=60, current_path = current_path)