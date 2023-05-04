import json
import os
from data.alfred_data import constants
from utils import plan_module
import pprint
from tqdm import tqdm

ACTIONS = ['PickupObject', 'PutObject', 'SliceObject', 'CleanObject', 'HeatObject', 'CoolObject', 'ToggleObject']
OBJECTS = (set(constants.OBJECTS) - constants.RECEPTACLES) | constants.MOVABLE_RECEPTACLES_SET
RECEPS = constants.RECEPTACLES & set(constants.OBJECTS) | {""}

result_path = 'data/available_actions2.json'
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

while len(triplets) != 0:
    _triplets = triplets[:buffer_size]
    triplets = triplets[buffer_size:]
    sentences, remove_idx = plan_module.get_action_description(_triplets)
    for i in remove_idx[::-1]:
        _triplets.pop(i)
        sentences.pop(i)
    new_data = dict(zip(sentences, _triplets))
    result_data = new_data.copy()
    triplets.extend([t for t in _triplets if t not in new_data.values()])
    # regenerate if sentence is duplicated
    for s in new_data:
        if s in avaiable_actions:
            triplets.append(avaiable_actions.pop(s))
            triplets.append(new_data[s])
            result_data.pop(s)
    avaiable_actions.update(result_data)
    json.dump(avaiable_actions, open(result_path, 'w'), indent=4)