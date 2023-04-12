Modified by Yuyeong Kim.

# Requirements
1. **openai api key** and **openai organization key** should reside in `utils/config.py` as `OPENAI` dictionary.
2. ALFRED data should reside in `data/alfred_data` folder.

# Preparation
## Generate Sentence described Plan data
~~~
python actions.py                           # transfer each triplet to sentence
python generate_plan_from_triplet.py        # match each triplet to sentence in train data
~~~

## Retrieve simialr plans from train data
~~~
pyhton retr_roberta.py
~~~
Retrive similar sentences or bit different sentences for example tasks.

# Generate Plan
## For each line
~~~
python plan_prompt-LineByLine.py
~~~
## Give a whole sequence
~~~
python plan_prompt.py
~~~