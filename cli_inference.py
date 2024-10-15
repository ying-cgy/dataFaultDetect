# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample
import os, sys
import pathlib
import pandas as pd
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

args = parser.parse_args()
cfg = Config(args)

model = SALMONN.from_config(cfg.config.model)
model.to(args.device)
model.eval()

wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)
# 运行python3 cli_inference.py --cfg-path configs/decode_config.yaml
# arr=["dog","rooster","pig","cow","frog","cat","hen","insects","sheep","crow"]
arr=["zero","one","two","three","four","five","six","seven","eight","nine"]
# arr=['anger','disgust ','fear ','happy ','neutral','sad']
def run(inPath,outPath):
    path = inPath #文件夹目录
    filesFreeSpoken= os.listdir(path) #得到文件夹下的所有文件名称
    aswcomlab=[]
    aswcomtname=[]
    aswcomcon=[]

    for i in filesFreeSpoken:
        print(i)
        comqes=[]
        str1 = i.split("_")
        # ESC
        # comqes.append("Describe the characteristic of the audio.")
        # comqes.append("What you hear in the audio?")
        # comqes.append("Can you hear a"+arr[int(str1[0])]+" on the audio?")
        # comqes.append("Is there a sound of "+arr[int(str1[0])]+" on the audio?")
        # comqes.append("Does the audio include a "+arr[int(str1[0])]+" ?")
        # comqes.append("Is the noise from "+arr[int(str1[0])]+" ?")

        # comqes.append("Can you hear the"+arr[int(str1[0])]+" on the audio?")
        # comqes.append("Is there a "+arr[int(str1[0])]+" on the audio?")
        # comqes.append("Does the audio sound like a "+arr[int(str1[0])]+" ?")
        # comqes.append("Can you identify the "+arr[int(str1[0])]+" from the audio??")

        #spoken digit
        comqes.append("Describe the context of the audio.")
        comqes.append("What you hear in the audio?")
        comqes.append("Is the audio contain '" + arr[int(str1[0])] + "' ?")
        comqes.append("Does the audio clearly pronounce '" + arr[int(str1[0])] + "' ?")
        comqes.append("Is the pronounciation of '" + arr[int(str1[0])] + "' distinct?")
        comqes.append("Does the audio contain the word '" + arr[int(str1[0])] + "' ?")

        # CREMAD
        # comqes.append("Describe the emotion of the audio.")
        # comqes.append("What emotion can you hear in the audio?")
        # comqes.append("Can you hear the person say in " + arr[int(str1[0])] + " emotion on the audio?")
        # comqes.append("Is someone speaking in a " + arr[int(str1[0])] + " sound on the audio?")
        # comqes.append("Does the speaker sound like " + arr[int(str1[0])] + " ?")
        # comqes.append("Can you hear a " + arr[int(str1[0])] + " emotion from the audio?")
        wav_path = path + i
        for q in comqes:
            try:
                print("====================================="+q)
                # print(wav_path)
                prompt = q
                samples = prepare_one_sample(wav_path, wav_processor)
                prompt = [
                    cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + prompt.strip())
                ]
                print("Output:")
                # for environment with cuda>=117
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output=model.generate(samples, cfg.config.generate, prompts=prompt)[0]
                    print(output)
                aswcomlab.append(str1[0])
                aswcomtname.append(i)
                aswcomcon.append(str(output))
                # print(model.generate(samples, cfg.config.generate, prompts=prompt)[0])
                # print(output, file=data)

            except Exception as e:
                print(e)
                import pdb; pdb.set_trace()

    aswcomframe = pd.DataFrame({'filename':aswcomtname,'label':aswcomlab,'answer':aswcomcon})
    aswcomframe.to_csv(outPath,sep=',')
# run("./dataset/ESC-50-master/noiseAudio/0.1/","./answer/ESC50_sym_answer.csv")
# run("./dataset/ESC-50-master/asyNoiseAudio/0.1/","./answer/ESC50_asym_answer.csv")
# run("./dataset/ESC-50-master/poisonAttack/","./answer/ESC50_poison_answer.csv")'

# run("./dataset/ESC-50-master/noiseAudio/0.4/","./answer/ESC50_sym_answer_4.csv")
# run("./dataset/ESC-50-master/asyNoiseAudio/0.4/","./answer/ESC50_asym_answer_4.csv")
# from answer_evaluation import answerEvaluation
# ans=answerEvaluation("./answer/ESC50_asym_answer_4.csv","./result/ESC50_asym_answer_4.csv")
# ans.evaluate()
run("./dataset/free-spoken-digit-dataset-1.0.9/asyNoiseAudio/0.4/","./answer/SpokenDigit_asym_answer_4.csv")
from answer_evaluation import answerEvaluation
ans=answerEvaluation("./answer/SpokenDigit_asym_answer_4.csv","./result/SpokenDigit_asym_answer_4.csv")
ans.evaluate()

# run("./dataset/free-spoken-digit-dataset-1.0.9/asyNoiseAudio/0.4/","./answer/SpokenDigit_asym_answer_4.csv")
# run("./dataset/free-spoken-digit-dataset-1.0.9/poisonAttack/","./answer/SpokenDigit_poison_answer_New.csv")

# run("/home/cgy/SALMONN/dataset/CREMAD/asyNoiseAudio/0.4/","./answer/CREMAD_asym_answer_4.csv")
# run("/home/cgy/SALMONN/dataset/CREMAD/asyNoiseAudio/0.1/","./answer/CREMAD_asym_answer.csv")
# run("/home/cgy/SALMONN/dataset/CREMAD/poisonAttack/","./answer/CREMAD_poison_answer.csv")

# from answer_evaluation import answerEvaluation
# ans=answerEvaluation("./answer/SpokenDigit_asym_answer_4.csv","./result/SpokenDigit_asym_answer_4.csv")
# ans.evaluate()

# ans=answerEvaluation("./answer/SpokenDigit_sym_answer_4.csv","./result/SpokenDigit_sym_answer_4.csv")
# ans.evaluate()
#
# ans=answerEvaluation("./answer/SpokenDigit_poison_answer.csv","./result/SpokenDigit_poison_answer.csv")
# ans.evaluate()# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#