import cv2
import tensorflow as tf
import numpy as np
import Config

def resizePic(img, width, height):
    retImage = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    return retImage

# 获取图片的width和length
def getWH(img):
    width = len(img[0])
    height = len(img)
    return width,height

def vec2Onehot(input, batch_size, Max_Len, Max_Width):
    '''
    将一个二维向量的第二个参数的值转换为one-hot向量集合
    :param input: 输入的数据的格式为二维向量 
    :param Max_Len: one-hot向量的最大长度，在本OCR项目中代表可以识别的最长字符长度
    :param Max_Width: one-hot向量的最大宽度，在本OCR项目中代表支持的字符个数
    :return: 返回形成的三维向量
    '''
    ret = np.zeros(shape=[batch_size,Max_Len,Max_Width])
    print('len(input)',len(input))
    print('len(input[0])', len(input[0]))
    for i in range(batch_size):
        st = str(input[i])
        for j in range(len(st)):
            for k in range(len(st)):
                ret[i][j][int(st[k])] = 1
    return ret

def onehot2Vec(onehot):
    ret = []
    temp = ''
    for i in range(len(onehot)):
        for j in range(len(onehot[i])):
            for k in range(len(onehot[i][j])):
                if (k == len(onehot[i][j])-1 and onehot[i][j][k] != 1):
                    j = len(onehot[i]) #默认字符的顺序是都在前边的，后边的都是0，出现了全0向量时，就认为是没有字符了
                if(onehot[i][j][k] == 1):
                    temp += Config.CH_DIC[onehot[i][j][k]]
                    k = len(onehot[i][j]) #one-hot向量中，找到为1的值后，就停止遍历当前向量
    if temp != '':
        ret.append(temp)
        temp = ''
    return ret


def vec2onehot(vec, batch_size = 64, Max_Len = 100, Max_Sup = 100):
    '''
    将batch的二维数组转换为三维的one-hot向量
    :param vec: vec是一个普通的二维数组
    :param batch_size: 
    :param Max_Len: 
    :param Max_Sup: 
    :return: 
    '''
    onehot = np.zeros(shape=[batch_size, Max_Len, Max_Sup])
    for i in range(batch_size):
        for j in range(Max_Len):
            onehot[i][j][Config.CH_DIC[vec[i][j]]] = 1
    return onehot

def vec2onehot_tensor(vec, batch_size = 64, Max_Len = 100, Max_Sup = 100):
    '''
    将batch的二维数组转换为三维的one-hot向量
    :param vec: 输入的vec为一个二维的tensor
    :param batch_size: 
    :param Max_Len: 
    :param Max_Sup: 
    :return: 
    '''
    vec = tf.cast(vec, dtype=tf.uint8)
    return tf.one_hot(vec, Max_Len, on_value=1.0, off_value=0.0, axis=-1)

