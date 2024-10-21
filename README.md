<meta name="robots" content="noindex">

This is the responsibility for paper " Leveraging Audio LLMs for Data Fault Detection in Audio Datasets"

We use audio LLMs to detect data fault. And we use [SALMONN](https://github.com/bytedance/SALMONN) as LLM.

If you want to use this, please download the [SALMONN](https://github.com/bytedance/SALMONN) and read its REMAND.md to set the environment.

``` python
# please change or add your questions and dataset in cli_inference.py, then run this file to get the answer.
python cli_inference.py
# you need to change the LLMApi.py, use your api and account. Then change the dataset and run this file.
python answer_evaluate.py

python metric_circulate.py
```
