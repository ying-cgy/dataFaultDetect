import pandas as pd
import numpy as np
from LLMApi import LLMApi

class answerEvaluation:
    # arrName = ["dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow"]
    # arrName = ['anger','disgust ','fear ','happy ','neutral','sad']
    arrName =["zero","one","two","three","four","five","six","seven","eight","nine"]
    answerPath=""
    resultPath=""
    def __init__(self,answerPath,resultPath):
        self.answerPath=answerPath
        self.resultPath=resultPath

    def calRate(self,arr,file):
        print(file,arr)
        ltrue=0
        lfalse=0
        for i in arr:
            if i==True:
                ltrue+=1
            else:
                lfalse+=1
        if ltrue>lfalse:
            return 0
        else:
            return 1
    def evaluate(self):

        common = pd.read_csv(self.answerPath)
        file1 = common['filename']
        realLabel=[]
        errArr = []
        api=LLMApi()

        file = file1.unique()
        index=0
        arrlist = np.zeros((6, len(file)))
        for ans in file:
            arr = []
            if(ans.split("_")[0]==ans.split("_")[1]):
                realLabel.append(0)
            else:realLabel.append(1)
            a=common[common['filename']==ans]
            # print(a['answer'])
            for i in range(len(a)):
                if i < 2:
                    # print(a['answer'][i+index*6])
                    api.context(a['answer'][i+index*6],self.arrName[a['label'][i+index*6]])
                    comans = api.urlCon()
                    arr.append('Yes' in comans)
                    arrlist[i][index]='Yes' in comans
                else:
                    arr.append('Yes' in a['answer'][i+index*6])
                    arrlist[i][index] = 'Yes' in a['answer'][i+index*6]
            errArr.append(self.calRate(arr, ans))
            index+=1
            # if index>=10:
            #     break
            # break
        # print("错误的标签个数有：" + len(errArr))
        aswcomframe = pd.DataFrame({'filename':file,'answer0':arrlist[0],'answer1':arrlist[1],'answer2':arrlist[2],'answer3':arrlist[3],
                                    'answer4':arrlist[4],'answer5':arrlist[5],'result':errArr,'realLabel':realLabel})
        aswcomframe.to_csv(self.resultPath,sep=',')
ans=answerEvaluation("./answer/SpokenDigit_asym_answer_4.csv","./result/SpokenDigit_asym_answer_4.csv")
ans.evaluate()
# ans=answerEvaluation("./answer/CREMAD_sym_answer.csv","./result/CREMAD_sym_answer.csv")
# ans.evaluate()
# ans=answerEvaluation("./answer/ESC50_sym_answer_4.csv","./result/ESC50_sym_answer_4.csv")
# ans.evaluate()
# ans=answerEvaluation("./answer/CREMAD_asym_answer.csv","./result/CREMAD_asym_answer.csv")
# ans.evaluate()
# ans=answerEvaluation("./answer/CREMAD_asym_answer_4.csv","./result/CREMAD_asym_answer_4.csv")
# ans.evaluate()
# ans=answerEvaluation("./answer/CREMAD_poison_answer.csv","./result/CREMAD_poison_answer.csv")
# ans.evaluate()
# ans=answerEvaluation("./answer/SpokenDigit_sym_answer_New.csv","./result/SpokenDigit_sym_answer_New.csv")
# ans.evaluate()
# ans=answerEvaluation("./answer/SpokenDigit_asym_answer_New.csv","./result/SpokenDigit_sym_answer_New.csv")
# ans.evaluate()
# ans=answerEvaluation("./answer/SpokenDigit_poison_answer_New.csv","./result/SpokenDigit_poison_answer_New.csv")
# ans.evaluate()