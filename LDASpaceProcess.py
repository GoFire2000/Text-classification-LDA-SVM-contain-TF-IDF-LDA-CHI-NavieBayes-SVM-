import os
import sys
import time
import pickle

from info_gain import info_gain
from sklearn.utils import Bunch
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from Tools import readFile, readBunchObj, writeBunchObj
from sklearn.decomposition import LatentDirichletAllocation

def fit_feature(feature_names,indics):
    rel_voc = []
    for i in indics:
        rel_voc.append(feature_names[i])
    return rel_voc

def vector_space(bunch_path, space_path, train_lda_path = None):
    print("开始LDA构建词模型...")
    bunch = readBunchObj(bunch_path)
    ldaSpace = Bunch(label = bunch.label, filenames = bunch.filenames, contents = bunch.contents, tdm = [], vocabulary = {})

    if train_lda_path is None:
        # 对训练集进行处理
        # vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5)
        # counts_train = vectorizer.fit_transform(bunch.contents)


        vectorizer = CountVectorizer()
        counts_train = vectorizer.fit_transform(bunch.contents)
        feature_names = vectorizer.get_feature_names()
        feature_dict = SelectKBest(chi2, k = 100)
        counts_train = feature_dict.fit_transform(counts_train, bunch.label)
        indics = feature_dict.get_support(indices=True)
        rel_voc = fit_feature(feature_names, indics)

        # ldaSpace.vocabulary = vectorizer.vocabulary_
        lda = LatentDirichletAllocation(n_components = 10, max_iter = 25, learning_method = 'batch')
        ldaSpace.tdm = lda.fit(counts_train).transform(counts_train)
    else:
        # 对测试集进行处理，利用训练集的信息
        # trainBunch = readBunchObj(train_lda_path)
        # ldaSpace.vocabulary = trainBunch.vocabulary

        vectorizer = CountVectorizer()
        counts_test = vectorizer.fit_transform(bunch.contents)
        feature_dict = SelectKBest(chi2, k = 100)
        # print(type(counts_test))
        # print(counts_test.shape)
        # print(type(bunch.label))
        # print(len(bunch.label))
        # counts_test = feature_dict.fit_transform(counts_test, bunch.label)
        vectorizer = TfidfVectorizer(sublinear_tf = True, max_df = 0.5)
        counts_test = vectorizer.fit_transform(bunch.contents)
        # ch2 = SelectKBest(chi2, k = 14)
        # counts_test = ch2.fit_transform(counts_test, bunch.label)
        # vectorizer = CountVectorizer(vocabulary = trainBunch.vocabulary)

        # counts_test = vectorizer.fit_transform(bunch.contents)
        lda = LatentDirichletAllocation(n_components = 10, max_iter = 25, learning_method = 'batch')
        ldaSpace.tdm = lda.fit(counts_test).transform(counts_test)
        print(ldaSpace.tdm.shape)

    writeBunchObj(space_path, ldaSpace)
    print("LDA构建词模型结束!")

if __name__ == '__main__':
    time1 = time.process_time()
    # 对训练集进行处理,LDA构建词模型
    bunch_path = "./BunchOfTrainingSets.dat"
    space_path = "./train_lda_spcae2.dat"
    vector_space(bunch_path, space_path)
    
    time2 = time.process_time()
    print('运行时间: %s s\n\n' % (time2 - time1))

    # 对测试集进行处理，LDA构建词模型，利用训练集的信息
    bunch_path = "./BunchOfTestSets.dat"
    space_path = "./test_lda_space2.dat"
    train_lda_path = "./train_lda_spcae2.dat"
    vector_space(bunch_path, space_path, train_lda_path)
    
    time3 = time.process_time()
    print('运行时间: %s s' % (time3 - time2))
    