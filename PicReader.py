import cv2
import os
import Config
from utils import *

lenth = 100
Max_Len = 100

def readPic(path):
    img = cv2.imread(path)
    width, height = getWH(img)
    return img, width, height

def getNextBatch(path_pic, num=64, Width=256, Height=256):
    batch_x = []
    batch_y = np.zeros(shape=[num, Max_Len])
    str_simple = []
    i = 0
    if os.path.exists(path_pic):
        dirs = os.listdir(path_pic)
        for dir in dirs:
            if not dir.endswith(Config.PICEND):
                continue
            img, width, height = readPic(path_pic + dir)
            img_re = resizePic(img, Width, Height)
            batch_x.append(img_re)
            text = dir[0:-4]
            str_simple.append(text)
            temp_str = str(text)
            for j in range(len(temp_str)):
                batch_y[i][j] = temp_str[j]
            i += 1

            print('==========================================================')
            print('i', i)
            print('batch_x', batch_x)
            print('batch_y', batch_y)
            print('dir', dir)
            if i>=num :
                break

    else:
        print('dir not exists')
    return batch_x, batch_y, str_simple

if __name__ == '__main__':
    batch_x, batch_y, str_simple = getNextBatch(Config.PICPATH)
    strs = onehot2Vec(batch_y)
    print(str(strs))