import os
if __name__ == '__main__':
    os.system('python CreateData.py') # 从主题.xlsx中提取出label和content，存到对应文件夹里(TrainingSets,TestSets)，方便查看
    os.system('python Pretreatment.py') # 分词，从上面文件夹中输出到新的Processed文件夹，分label和unlabel
    os.system('python SetsToBunchProcess.py') # 从分词后文件夹保存到bunch中，格式为dat，加快io速度
    os.system('python LDASpcaeProcess.py') # LDA，注释和保留的分别为infoGain、chi类型
    os.system('python SupportVectorMachinePredict.py') # 把LDA得到的训练和测试
    os.system('python resultOutput.py') # 输出成图片