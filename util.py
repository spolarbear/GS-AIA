# 2017 . 10 . 19
# Copyright GDADRI
# 版权归广东省建筑设计研究院所有
# Author Yangxin
# 作者：杨新
# Contact / spolarbear@qq.com
# 联系邮箱 spolarbear@qq.com

# 基于tensorflow的智能设计
# 读取样本集像素内容

from glob import glob
import os
from PIL import Image 
from matplotlib import pyplot as plt  
import scipy.misc
from scipy.ndimage import filters
import numpy as np
import random
from PIL import ImageChops


class gdad_init:
    def __init__(self,w=60,h=60):
        self.w = w
        self.h = h
        self.sample_size = 20

        self.ifFlatten = False

        self.data,self.label=self.read_img("D:/tensorflow/rect")
        self.data = np.asarray(self.data,np.float32)
        #打乱顺序
        self.num_example=self.data.shape[0]
        train_size = self.num_example

        index = [i for i in range(len(self.data))] 
        random.shuffle(index)
        dataRAM=[]
        labelRAM=[]
        for i in range(len(self.data)):
            dataRAM.append(self.data[index[i]])
            labelRAM.append(self.label[index[i]])

        #将所有数据分为训练集和验证集
        ratio=1
        s=np.int(self.num_example*ratio)
        x_train=dataRAM[:s]
        y_train=labelRAM[:s]
        x_val=dataRAM[s:]
        y_val=labelRAM[s:]


        self.X = np.asarray(x_train,np.float32) 
        self.Y = np.asarray(y_train,np.float32) 
        self.testX =  np.asarray(x_val,np.float32) 
        self.testY =  np.asarray(y_val,np.float32) 

    def show_image_matrix(self, id):
        d = self.testX[id]#.reshape((32,32))
        # print(d)
        imgssss = self.MatrixToImage(d)
        print(self.testY[id])
        self.show_img(imgssss)
    def getXY(self):
        return self.X,self.Y
    def getTestXY(self):
        return self.testX,self.testY
    def show_img(self, img):
        fig = plt.figure()  
        ax = fig.add_subplot(121)  
        ax.imshow(img)  
        plt.show()#显示刚才所画的所有操作 
    def dense_to_one_hot(self, num_labels, num_classes=10):
        labels_one_hot = np.zeros(num_classes)
        labels_one_hot[num_labels] = 1
        return labels_one_hot

    def appendImg(self,img,label):
        if(self.ifFlatten):
             img = img.ravel()
        img = np.asarray(img,np.float32)/127.5 - 1
        self.imgs.append(img)
        apimid = self.dense_to_one_hot(label,self.sample_size)
        self.labels.append(apimid)
    def read_img(self, path):

        cate = glob(os.path.join(path, "*.jpg"))
        self.imgs=[]
        self.labels=[]
        imid=0
        for idx,filename in enumerate(cate):
            img=Image.open(filename)
            imid=filename.replace(path,"").replace(".jpg","").replace("\\","")
            imid = int(imid)
            img = img.resize((self.w , self.h)) # 改变大小  
            #rotate
            i=0
            while i<360:
                img0 = img.rotate(i) #旋转  
                self.appendImg(img0,imid)
                i=i+90
                #offset
                offset = 1
                while offset<4:
                    img11  = ImageChops.offset(img0, offset, 0)
                    img12  = ImageChops.offset(img0, 0, offset)
                    self.appendImg(img11,imid)
                    self.appendImg(img12,imid)
                    offset += 1
                    #gaussian
                    j=0.1
                    while j<0.5:
                        #print(imid)
                        imgGS  = filters.gaussian_filter(img11,j)
                        j+=0.1
                        self.appendImg(imgGS,imid)
                        # if(imid==4):
                        #     show_img(imgFill)
                        #     print(apimid)
        return self.imgs, self.labels
    def MatrixToImage(self, data):
        data = data*255
        new_im = Image.fromarray(data.astype(np.uint8))
        return new_im


# gdad = gdad_init()
# X, Y = gdad.getXY()
# testX, testY = gdad.getTestXY()
# sample_size = gdad.sample_size

# if(True):  #  show the image
#     gdad.show_image_matrix(0)
