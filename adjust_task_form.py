import argparse
import os
import json
import torch
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import parmap
from data.alfred_data import constants

GPU_USED = [0]

OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]

MOVABLE_RECEPTACLES = [
    'Bowl',
    'Box',
    'Cup',
    'Mug',
    'Plate',
    'Pan',
    'Pot',
]

RECEPTACLES = {
        'BathtubBasin',
        'Bowl',
        'Cup',
        'Drawer',
        'Mug',
        'Plate',
        'Shelf',
        'SinkBasin',
        'Box',
        'Cabinet',
        'CoffeeMachine',
        'CounterTop',
        'Fridge',
        'GarbageCan',
        'HandTowelHolder',
        'Microwave',
        'PaintingHanger',
        'Pan',
        'Pot',
        'StoveBurner',
        'DiningTable',
        'CoffeeTable',
        'SideTable',
        'ToiletPaperHanger',
        'TowelHolder',
        'Safe',
        'BathtubBasin',
        'ArmChair',
        'Toilet',
        'Sofa',
        'Ottoman',
        'Dresser',
        'LaundryHamper',
        'Desk',
        'Bed',
        'Cart',
        'TVStand',
        'Toaster',
    }

def preprocess(s):
    # remove escape sequence
    if "\n" in s:
        s = s.split("\n")[0]
    elif "\b" in s:
        s = s.strip("\b")
    # remove number in word
    s = re.sub(r'[0-9]+', '', s)
    # CounterTop -> Counter Top
    if not any(_s.isupper() for _s in s):
        return s.lower()
    if s == 'TVStand':
        return 'tv stand'
    if s == 'TVDrawer':
        return 'tv drawer'
    if s == 'TVRemote':
        return 'tv remote'
    if s.upper() == s:
        return s.lower()
    l = re.findall('[A-Z][^A-Z]*', s)
    ls = []
    left = s.split(l[0])[0]
    for _l in l:
        if len(_l) == 1:
            left += _l
        else:
            _l = left+_l
            ls.extend(_l.split())
            left = ''
    # lower
    return ' '.join(ls).lower().replace('.', '')

def closest_object_roberta(key, set):
    set = list(set)
    oList = [preprocess(o) for o in set]
    gpuId = (GPU_USED[0]+1)%1
    GPU_USED[0] = gpuId
    device = torch.device("cuda:%d"%gpuId if torch.cuda.is_available() else "cpu")
    translation_lm = SentenceTransformer('stsb-roberta-large').to(device)
    example_task_embedding = translation_lm.encode(oList, batch_size=512, convert_to_tensor=True, device=device)
    query_embedding = translation_lm.encode(preprocess(key), convert_to_tensor=True, device=device)
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, example_task_embedding)[0].detach().cpu().numpy()
    torch.cuda.empty_cache()
    idx = np.argsort(cos_scores)[-1]
    most_similar_object, matching_score = set[idx], cos_scores[idx]
    return most_similar_object, matching_score

def adjust_param(param):
    new_param = {}
    # 1. task type에 빈칸 정리하기
    if param['task_type'] == 'pick_and_place_with_movable_recep' \
        and param['mrecep_target'] == '':
        new_param['task_type'] = 'pick_and_place_simple'

    # 2. object out of range error
    if param['object_target'] not in OBJECTS:
        replaced_object, _ = closest_object_roberta(param['object_target'], set(OBJECTS))
        new_param['object_target'] = replaced_object
    if param['parent_target'] != '' and param['parent_target'] not in RECEPTACLES:
        replaced_object, _ = closest_object_roberta(param['parent_target'], RECEPTACLES)
        new_param['parent_target'] = replaced_object
    if param['mrecep_target'] != '' and param['mrecep_target'] not in MOVABLE_RECEPTACLES:
        replaced_object, _ = closest_object_roberta(param['mrecep_target'], set(MOVABLE_RECEPTACLES))
        new_param['mrecep_target'] = replaced_object
    if param['toggle_target'] != '' and param['toggle_target'] not in ['FloorLamp', 'DeskLamp']:
        replaced_object, _ = closest_object_roberta(param['toggle_target'], set(['FloorLamp', 'DeskLamp']))
        new_param['toggle_target'] = replaced_object

    for key in param:
        if key not in new_param:
            new_param[key] = param[key]
    return new_param

def main(args):
    if not os.path.exists(args.json_file):
        raise Exception(FileNotFoundError)
    match = json.load(open(args.json_file, 'r'))
    goal = list(match.keys())
    ret = parmap.map(adjust_param, [match[g][0] for g in goal], pm_pbar=True, pm_processes=4)
    replaced = dict(zip(goal, ret))
    write_json = '-replaced.'.join(args.json_file.split('.'))
    print('Save in file {}'.format(write_json))
    json.dump(replaced, open(write_json, 'w'), indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file', type=str)
    args = parser.parse_args()
    main(args)