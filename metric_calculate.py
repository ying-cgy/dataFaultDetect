import numpy as np
# from sklearn import metric
import pandas as pd
common = pd.read_csv('/home/cgy/SALMONN/result/SpokenDigit_asym_answer_4.csv')

realLabel=common['realLabel']
# result=common['result']
answer=[common['answer0'],common['answer1'],common['answer2'],common['answer3'],common['answer4'],common['answer5']]

def cal(threshold):

    result=[]
    for i in range(len(realLabel)):
        ltrue=0
        for j in range(6):
            if answer[j][i]==1.0:
                ltrue+=1
        # print(ltrue)
        if ltrue/6 > threshold:
            result.append(0)
        else:
            result.append(1)
        # break
    # print(len(result),len(realLabel))
    from sklearn.metrics import precision_score, recall_score,confusion_matrix
    from sklearn import metrics
    # precision = precision_score(realLabel, result)
    # recall = recall_score(realLabel, result)
    # print(f'Precision: {precision}, Recall: {recall}')\
    fpr, tpr, _ = metrics.roc_curve(realLabel,result)
    print("metric:",tpr,fpr)

    roc = metrics.auc(fpr, tpr)
    ap = metrics.average_precision_score(realLabel,result)

    print(roc,ap)
    return tpr,fpr
threshold=[0.167,0.334,0.5,0.667,0.834]
for i in threshold:
    tpr,fpr=cal(i)
    # print(i,tpr,fpr)
