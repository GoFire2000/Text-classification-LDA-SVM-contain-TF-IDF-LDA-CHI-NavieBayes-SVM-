执行顺序
os.system('python CreateData.py') # 从主题.xlsx中提取出label和content，存到对应文件夹里(TrainingSets,TestSets)，方便查看
os.system('python Pretreatment.py') # 分词，从上面文件夹中输出到新的Processed文件夹，分label和unlabel
os.system('python SetsToBunchProcess.py') # 从分词后文件夹保存到bunch中，格式为dat，加快io速度
os.system('python LDASpcaeProcess.py') # LDA，注释和保留的分别为infoGain、chi类型
os.system('python SupportVectorMachinePredict.py') # 把LDA得到的训练和测试
os.system('python resultOutput.py') # 输出成图片

注意，对应不同类型的路径，比如chi+label、chi+unlabel、infogain+label，需要改变代码里的文件路径（都在main函数中），在改变chi和infogain时，需要去LDASpaceProcess中调整注释内容

三个csv是三条路径的的结果
png是图片结果

顺便，我保留了tfidf的99%准确率，需要LDASpaceProcess改为TF-IDFSpaceProcess，后面的分类器代码都加了个2，比如SupportSvectorMacheine2.py

不推荐每次都执行main.py，费时间
最好一个一个执行