import os
import json
import re

original_data_path = 'alfred/data/json_2.1.0/train'
aciton_path = 'language-planner/data/available_actions.json'
examples_data_path = 'language-planner/data/available_examples.json'

actionList = []
examples = []
for ep in os.listdir(original_data_path):
    for trial in os.listdir(os.path.join(original_data_path, ep)):
        with open(os.path.join(original_data_path, ep, trial, 'traj_data.json'), 'r') as f:
            data = json.load(f)
        annList = data['turk_annotations']['anns']
        # vote 수에 따른 filtering 필요할까?
        # 이 예시에서 executable action들은 더욱 정형화되어있고, 문장이 하나다. (and 같은 걸로 이어져 있지 않음)
        for ann in annList:
            actions = [re.sub("r[^a-z0-9]", "", a.lower()) for a in ann['high_descs']]
            actionList.extend(actions)
            task = ann['task_desc']
            step = ''
            for i, action in enumerate(ann['high_descs']):
                step += 'Step {}: {}\n'.format(i+1, action)
            ex = 'Task: {task}\n{step}'.format(task=task, step=step[:-2])
            examples.append(ex)
with open(aciton_path, 'w+') as f:
    json.dump(actionList, f)
with open(examples_data_path, 'w+') as f:
    json.dump(examples, f)

