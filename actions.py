import json
from data.alfred_data import constants
from utils import plan_module
import pprint
from tqdm import tqdm

ACTIONS = ['PickupObject', 'PutObject', 'CleanObject', 'HeatObject', 'CoolObject', 'ToggleObject', 'SliceObject']
OBJECTS = constants.OBJECTS
RECEPS = constants.RECEPTACLES

result_path = 'data/available_actions1.json'

avaiable_actions = {}
triplets = []
pp = pprint.PrettyPrinter()
for action in ACTIONS:
    for obj in OBJECTS:
        for recep in RECEPS:
            triplet = [action, obj, recep]
            triplets.append(triplet)

len_triplets = len(triplets)
buffer_size = 20 # num of parallel request
for i in tqdm(range(int(len_triplets/buffer_size)+1)):
    _triplets = triplets[buffer_size*i:buffer_size*(i+1)]
    sentences = plan_module.get_action_description(_triplets)
    avaiable_actions.update(dict(zip(sentences, _triplets)))
    
json.dump(avaiable_actions, open(result_path, 'w'), indent=4)