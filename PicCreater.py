import cv2
import numpy as np
import random
from utils import *


Max_Len = 100
Max_Width = 100

def create(text, height = 32, width = 256, rgb = 3, textColor = (0,0xff,0), backColor = 0xff):
    img = np.zeros((height, width, rgb), np.uint8)
    cv2.putText(img, text, (0, height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor)
    return img

def saveImg(img, path):
    cv2.imwrite(path, img)

def getNextBatchMem(lenth=64, width=256, height=256):
    batch_x = []
    batch_y = np.zeros(shape=[lenth, Max_Len])
    str_simple = []
    for i in range(lenth):
        temp_num = random.randint(1, 1000)
        img = create(str(temp_num), rgb=1, textColor=(0,0xff,0))
        img = resizePic(img, width, height)
        batch_x.append(img)
        str_simple.append(temp_num)
        temp_str = str(temp_num)
        for j in range(len(temp_str)):
            batch_y[i][j] = temp_str[j]
    return batch_x, batch_y, str_simple

if __name__ == '__main__':
    for i in range(500):
        rand_text = str(random.randint(1000, 123213))
        image = create(rand_text)
        saveImg(image, 'D:\\pic\\' + rand_text + '.jpg')