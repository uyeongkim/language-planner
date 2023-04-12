Modified by Yuyeong Kim.

# Requirement
1. **openai api key** and **openai organization key** should reside in `utils/config.py` as `OPENAI` dictionary.
2. ALFRED data should reside in `data/alfred_data` folder.

# Preparation
## Generate Available actions
~~~
python actions.py
~~~
## Retrieve simialr plans from train data
~~~
pyhton retr_roberta.py
~~~


# Generate Plan
## For each line
~~~
python plan_prompt-LineByLine.py
~~~
## Give a whole sequence
~~~
python plan_prompt.py
~~~