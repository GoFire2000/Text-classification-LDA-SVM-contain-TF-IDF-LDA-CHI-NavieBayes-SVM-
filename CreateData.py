
import os
import time
import jieba
import numpy as np
import pandas as pd
# import sklearn
from sklearn.model_selection import train_test_split
from Tools import saveFile, readFile

def CreateDataSet(initial_path, dataX, dataY, newDataX = [], newDataY = [], tag = 0):
    if not os.path.exists(initial_path):
        os.makedirs(initial_path)
    dir_list = os.listdir(initial_path)

    L, R = 0, 0
    dataY.append('NONE')
    idx = 0
    for i in range(len(dataY)):
        if i == 0:
            L = i
        else:
            if dataY[i] != dataY[i - 1]:
                R = i - 1
                ini_path = initial_path + dataY[i - 1] + '/'
                if not os.path.exists(ini_path):
                    os.makedirs(ini_path) 
                file_list = os.listdir(ini_path)
                num = 0
                content = ''
                for j in range(L, R + 1):
                    num += 1
                    content = content + dataX[j]
                    content = content + '\n'
                    if num == 5 or j == R:
                        fullpath = ini_path + 'data' + str(idx) + '.txt'
                        idx += 1
                        if len(content) > 50:
                            saveFile(fullpath, content.encode('utf-8'))
                            if tag == 1:
                                newDataX.append(content)
                                newDataY.append(dataY[i - 1])
                            content = ''
                            num = 0
                    
                    # print(fullpath)
                
                L = i
    dataY.pop()

if __name__ == '__main__':
    time1 = time.process_time()
    print('从主题.xlsx获取源数据...')
    dataAll = pd.DataFrame(pd.read_excel('./主题.xlsx'))
    dataX = []
    dataY = []
    
    for i in range(len(dataAll.index)):
        # print(i)
        dataX.append(dataAll.iloc[i, 1])
        dataY.append(dataAll.iloc[i, 0])
        if type(dataY[-1]) != type('123'): # 去掉空的情况
            dataX.pop()
            dataY.pop()
            continue
        if type(dataX[-1]) != type('123'):
            dataX.pop()
            dataY.pop()
            continue
        if len(dataX[-1]) < 20:
            dataX.pop()
            dataY.pop()
    
    print(len(dataAll.index))
    print(len(dataX))
    newDataX = []
    newDataY = []

    time2 = time.process_time()
    print('运行时间: %s s\n\n' % (time2 - time1))
    
    print('创建训练集合数据...')
    initial_sets_path = "./TrainingSets/" # 未分词的训练集路径
    CreateDataSet(initial_sets_path, dataX, dataY, newDataX, newDataY, 1)
    
    time3 = time.process_time()
    print('运行时间: %s s' % (time3 - time2))

    
    x_train, x_test, y_train, y_test = train_test_split(newDataX, newDataY, test_size = 0.3)
    
    print('创建测试数据集合...')
    initial_sets_path = "./TestSets/" # 未分词的测试集路径
    CreateDataSet(initial_sets_path, x_test, y_test)
    time4 = time.process_time()
    print('运行时间: %s s' % (time4 - time3))
    
    
