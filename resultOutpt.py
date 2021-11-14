import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def getValue(path):
    df = pd.read_csv(path)
    # 正确分类率，错误分类率，平均正确率，平均召回率，平均F值
    v1 = df.iloc[-3][1]
    v2 = 1.0 - v1
    v3 = df['precision'][0: -3].sum() / len(df.index - 3)
    v4 = df['recall'][0: -3].sum() / (len(df.index) - 3)
    v5 = df['f1-score'][0: -3].sum() / (len(df.index) - 3)
    return v1, v2, v3, v4, v5

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom')

if __name__ == '__main__':
    label_chi = getValue('./label_chi2_lda_result.csv')
    label_infogain = getValue('./label_infogain_lda_result.csv')
    unlabel_chi = getValue('./unlabel_chi_lda_result.csv')

    labels = ['正确分类率', '错误分类率', '平均正确率', '平均召回率', '平均F值']
    lis = [[] for i in range(3)]
    
    for i in range(5):
        lis[0].append(label_chi[i])
        lis[1].append(label_infogain[i])
        lis[2].append(unlabel_chi[i])

    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    x = np.arange(len(labels))

    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    rects1 = ax.bar(x - width*2, lis[0], width, label='label_chi')
    rects2 = ax.bar(x - width+0.04, lis[1], width, label='label_infoGain')
    rects3 = ax.bar(x + 0.08, lis[2], width, label='unlabel_chi')


    # 为y轴、标题和x轴等添加一些文本。
    # ax.set_ylabel('Y轴', fontsize=16)
    # ax.set_xlabel('X轴', fontsize=16)
    ax.set_title('data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()




    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    fig.tight_layout()
    plt.savefig('result.png')
    plt.show()
        

    
