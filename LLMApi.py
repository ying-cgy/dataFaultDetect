import requests
import json

API_KEY = "qpBA3KSw2jhdOvs1DQ4kg7im"
SECRET_KEY = "TBdsZRto3kRalDUDyNlWHTV3fZqUPiHV"

class LLMApi:
    # def __init__(self, text):
    #     self.text = text
    content=''
    label=''
    def context(self,t,l):
        self.content=t
        self.label=l
    def urlCon(self):
        #7.18下线，可改为https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-lite-8k
        # url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant?access_token=" + self.get_access_token()
        url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-speed-128k?access_token=" + self.get_access_token()

        payload = json.dumps({
            "messages": [
                {
                    "role": "user",
                    # "content": "Assume you are a helpful and precise assistant for evaluation. Please judge"
                    #            "whether the ‘Caption’ of an audio and one of the ‘Label’ refer to the same"
                    #            "object. Answer with yes or no. For example, Caption:A rooster crowing. Label: Rooster. And the answer is yes."
                    #            "Another example: Caption:The characteristic of the audio is a buzzing bee sound. Label: Insects. And the answer is yes."
                    #            "- Caption: "+self.content+
                    #            "- Label:"+self.label
                    # "content": "Assume you are a helpful and precise assistant for evaluation. Please judge"
                    #            "whether the ‘Caption’ of an audio and one of the ‘Label’ refer to the same"
                    #            "emotion. Answer with yes or no. For example, Caption:The emotion of the audio is calm and reassuring. Label: neutral. And the answer is yes."
                    #            "Another example: Caption:I heard sadness in the audio. Label: sad. And the answer is yes."
                    #            "- Caption: " + self.content +
                    #            "- Label:" + self.label
                    "content": "Assume you are a helpful and precise assistant for evaluation. Please judge"
                               "whether the ‘Caption’ of an audio and one of the ‘Label’ refer to the same"
                               "word. Answer with yes or no. For example, Caption:The audio is of a male voice saying the number nine. Label: nine. And the answer is yes."
                               "Another example: Caption:Someone pronouncing the number 'four'. Label: four. And the answer is yes."
                               "- Caption: " + self.content +
                               "- Label:" + self.label

                }
            ],
            # "temperature": 0.95,
            # "top_p": 0.7,
            # "penalty_score": 1
        })
        headers = {
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        # print(response.text)
        # print(type(response.text))
        json_data = json.loads(response.text)
        # print(json_data)
        return json_data['result']


    def get_access_token(self,):
        """
        使用 AK，SK 生成鉴权签名（Access Token）
        :return: access_token，或是None(如果错误)
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))

#
# if __name__ == '__main__':
#     main(text)
