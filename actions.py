import json
import os
from data.alfred_data import constants
from utils import plan_module
import pprint
from tqdm import tqdm

ACTIONS = ['PickupObject', 'PutObject', 'SliceObject', 'CleanObject', 'HeatObject', 'CoolObject', 'ToggleObject']
OBJECTS = (set(constants.OBJECTS) - constants.RECEPTACLES) | constants.MOVABLE_RECEPTACLES_SET
RECEPS = (constants.RECEPTACLES | {""}) & set(constants.OBJECTS)

result_path = 'data/available_actions1.json'
buffer_size = 20 # num of parallel request

if os.path.exists(result_path):
    avaiable_actions = json.load(open(result_path, 'r'))
else:
    avaiable_actions = {}

triplets = []
pp = pprint.PrettyPrinter()
for action in ACTIONS:
    for obj in OBJECTS:
        if action in ['PickupObject', 'PutObject', 'SliceObject']:
            for recep in RECEPS:
                if action == 'PutObject' and recep == '':
                    continue
                triplet = [action, obj, recep]
                if triplet not in avaiable_actions.values():
                    triplets.append(triplet)
        else:
            if action == 'CleanObject':
                triplet = [action, obj, 'SinkBasin']
            elif action == 'HeatObject':
                triplet = [action, obj, 'Microwave']
            elif action == 'CoolObject':
                triplet = [action, obj, 'Fridge']
            else:
                triplet = [action, obj, obj]

            if triplet not in avaiable_actions.values():
                triplets.append(triplet)

len_triplets = len(triplets)
checkpoint = 0
for i in tqdm(range(int(len_triplets/buffer_size)+1)):
    _triplets = triplets[buffer_size*i:buffer_size*(i+1)]
    sentences = plan_module.get_action_description(_triplets)
    avaiable_actions.update(dict(zip(sentences, _triplets)))
    if len(avaiable_actions) > checkpoint * 1000:
        checkpoint += 1
        json.dump(avaiable_actions, open(result_path, 'w'), indent=4)
    
json.dump(avaiable_actions, open(result_path, 'w'), indent=4)