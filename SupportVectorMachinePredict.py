import os
import time
import pickle
import pandas as pd

from sklearn import metrics
from Tools import readBunchObj
from sklearn.naive_bayes import MultinomialNB # 导入多项式贝叶斯算法
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix

if __name__ == "__main__":
    time1 = time.process_time()

    # 导入训练集
    train_lda_path = "./train_lda_spcae2.dat"
    train_lda_sets = readBunchObj(train_lda_path)
    
    # 导入测试集
    test_lda_path = "./test_lda_space2.dat"
    test_lda_sets = readBunchObj(test_lda_path)

    # 训练分类器，输入词袋向量和分类标签
    # 朴素贝叶斯，NaiveBayes
    # classifier = MultinomialNB(alpha = 0.000001).fit(train_lda_sets.tdm, train_lda_sets.label)
    
    # Support Vector Machin
    # classifier = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
    # classifier.fit(train_lda_sets.tdm, train_lda_sets.label)
    classifier = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    classifier.fit(train_lda_sets.tdm, train_lda_sets.label)
    # 预测分析结果
    predictedResult = classifier.predict(test_lda_sets.tdm)
    
    # 错误信息
    # print("错误信息如下:")
    # for act_label, file_name, expct_label in zip(test_lda_sets.label, test_lda_sets.filenames, predictedResult):
    #     if act_label != expct_label:
    #         print(file_name, ": 实际类别:", act_label, "   预测类别:", expct_label)

    print("\n\n每类和总体正确率、召回率、f1-score如下")
    print(classification_report(test_lda_sets.label, predictedResult))

    out = classification_report(test_lda_sets.label, predictedResult, output_dict = True)
    df = pd.DataFrame(out).transpose()
    df.to_csv('unlabel_chi_lda_result.csv',index = True)

    time2 = time.process_time()    
    print('运行时间: %s s' % (time2 - time1))

