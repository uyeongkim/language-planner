import json
import random 
import re
import openai
import time
import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils

# Yonsei api key
openai.api_key = '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'
openai.organization = 'org-azdthpxrguDHQc2ujvxf4hTZ'

# yuyeong personal api key
# openai.api_key = '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'
# openai.organization = 'org-pLk6QBv25E0v68rzU9KfJXiu'

OBJECTS = ['AlarmClock','Apple','ArmChair','BaseballBat','BasketBall','Bathtub','BathtubBasin','Bed','Blinds','Book','Boots','Bowl','Box','Bread','ButterKnife','Cabinet','Candle','Cart','CD','CellPhone','Chair','Cloth','CoffeeMachine','CounterTop','CreditCard','Cup','Curtains','Desk','DeskLamp','DishSponge','Drawer','Dresser','Egg','FloorLamp','Footstool','Fork','Fridge','GarbageCan','Glassbottle','HandTowel','HandTowelHolder','HousePlant','Kettle','KeyChain','Knife','Ladle','Laptop','LaundryHamper','LaundryHamperLid','Lettuce','LightSwitch','Microwave','Mirror','Mug','Newspaper','Ottoman','Painting','Pan','PaperTowel','PaperTowelRoll','Pen','Pencil','PepperShaker','Pillow','Plate','Plunger','Poster','Pot','Potato','RemoteControl','Safe','SaltShaker','ScrubBrush','Shelf','ShowerDoor','ShowerGlass','Sink','SinkBasin','SoapBar','SoapBottle','Sofa','Spatula','Spoon','SprayBottle','Statue','StoveBurner','StoveKnob','DiningTable','CoffeeTable','SideTable','TeddyBear','Television','TennisRacket','TissueBox','Toaster','Toilet','ToiletPaper','ToiletPaperHanger','ToiletPaperRoll','Tomato','Towel','TowelHolder','TVStand','Vase','Watch','WateringCan','Window','WineBottle']
VAL_RECEPTACLE_OBJECTS = {
    'Pot': {'Apple','ButterKnife','DishSponge','Egg','Fork','Knife','Ladle','Lettuce','LettuceSliced','Potato','PotatoSliced','Spatula','Spoon','Tomato','TomatoSliced'},
    'Pan': {'Apple',
            'ButterKnife',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'LettuceSliced',
            'Potato',
            'PotatoSliced',
            'Spatula',
            'Spoon',
            'Tomato',
            'TomatoSliced'},
    'Bowl': {'Apple',
            'ButterKnife',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'LettuceSliced',
            'Potato',
            'PotatoSliced',
            'Spatula',
            'Spoon',
            'Tomato',
            'TomatoSliced',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'DishSponge',
            'KeyChain',
            'Mug',
            'PaperTowel',
            'Pen',
            'Pencil',
            'RemoteControl',
            'Watch'},
    'CoffeeMachine': {'Mug'},
    'Microwave': {'Apple',
                'Bowl',
                'Bread',
                'BreadSliced',
                'Cup',
                'Egg',
                'Glassbottle',
                'Mug',
                'Plate',
                'Potato',
                'PotatoSliced',
                'Tomato',
                'TomatoSliced'},
    'StoveBurner': {'Kettle',
                    'Pan',
                    'Pot'},
    'Fridge': {'Apple',
            'Bowl',
            'Bread',
            'BreadSliced',
            'Cup',
            'Egg',
            'Glassbottle',
            'Lettuce',
            'LettuceSliced',
            'Mug',
            'Pan',
            'Plate',
            'Pot',
            'Potato',
            'PotatoSliced',
            'Tomato',
            'TomatoSliced',
            # Not in alfred constants but in gt
            'Knife',
            'ButterKnife'
            ###
            'WineBottle'},
    'Mug': {'ButterKnife',
            'Fork',
            'Knife',
            'Pen',
            'Pencil',
            'Spoon',
            'KeyChain',
            'Watch'},
    'Plate': {'Apple',
            'ButterKnife',
            'DishSponge',
            'Egg',
            'Fork',
            'Knife',
            'Ladle',
            'Lettuce',
            'LettuceSliced',
            'Mug',
            'Potato',
            'PotatoSliced',
            'Spatula',
            'Spoon',
            'Tomato',
            'TomatoSliced',
            'AlarmClock',
            'Book',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'DishSponge',
            'Glassbottle',
            'KeyChain',
            'Mug',
            'PaperTowel',
            'Pen',
            'Pencil',
            'TissueBox',
            'Watch'},
    'Cup': {'ButterKnife',
            'Fork',
            'Spoon'},
    'Sofa': {'BasketBall',
            'Book',
            'Box',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'KeyChain',
            'Laptop',
            'Newspaper',
            'Pillow',
            'RemoteControl'},
    'ArmChair': {'BasketBall',
                'Book',
                'Box',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'KeyChain',
                'Laptop',
                'Newspaper',
                'Pillow',
                'RemoteControl'},
    'Box': {'AlarmClock',
            'Book',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'DishSponge',
            'Glassbottle',
            'KeyChain',
            'Mug',
            'PaperTowel',
            'Pen',
            'Pencil',
            'RemoteControl',
            'Statue',
            'TissueBox',
            'Vase',
            'Watch'},
    'Ottoman': {'BasketBall',
                'Book',
                'Box',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'KeyChain',
                'Laptop',
                'Newspaper',
                'Pillow',
                'RemoteControl'},
    'Dresser': {'AlarmClock',
                'BasketBall',
                'Book',
                'Bowl',
                'Box',
                'Candle',
                'CD',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'Cup',
                'Glassbottle',
                'KeyChain',
                'Laptop',
                'Mug',
                'Newspaper',
                'Pen',
                'Pencil',
                'Plate',
                'RemoteControl',
                'SprayBottle',
                'Statue',
                'TennisRacket',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Vase',
                'Watch',
                'WateringCan',
                'WineBottle'},
    'LaundryHamper': {'Cloth'},
    'Desk': {'AlarmClock',
            'BasketBall',
            'Book',
            'Bowl',
            'Box',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'Cup',
            'Glassbottle',
            'KeyChain',
            'Laptop',
            'Mug',
            'Newspaper',
            'Pen',
            'Pencil',
            'Plate',
            'RemoteControl',
            'SoapBottle',
            'SprayBottle',
            'Statue',
            'TennisRacket',
            'TissueBox',
            'ToiletPaper',
            'ToiletPaperRoll',
            'Vase',
            'Watch',
            'WateringCan',
            'WineBottle'},
    'Bed': {'BaseballBat',
            'BasketBall',
            'Book',
            'CellPhone',
            'Laptop',
            'Newspaper',
            'Pillow',
            'TennisRacket'},
    'Toilet': {'Candle',
            'Cloth',
            'DishSponge',
            'Newspaper',
            'PaperTowel',
            'SoapBar',
            'SoapBottle',
            'SprayBottle',
            'TissueBox',
            'ToiletPaper',
            'ToiletPaperRoll',
            'HandTowel'},
    'ToiletPaperHanger': {'ToiletPaper',
                        'ToiletPaperRoll'},
    'TowelHolder': {'Towel'},
    'HandTowelHolder': {'HandTowel'},
    'Cart': {'Candle',
            'Cloth',
            'DishSponge',
            'Mug',
            'PaperTowel',
            'Plunger',
            'SoapBar',
            'SoapBottle',
            'SprayBottle',
            'Statue',
            'TissueBox',
            'ToiletPaper',
            'ToiletPaperRoll',
            'Vase',
            'HandTowel'},
    'BathtubBasin': {'Cloth',
                    'DishSponge',
                    'SoapBar',
                    'HandTowel'},
    'SinkBasin': {'Apple',
                'Bowl',
                'ButterKnife',
                'Cloth',
                'Cup',
                'DishSponge',
                'Egg',
                'Glassbottle',
                'Fork',
                'Kettle',
                'Knife',
                'Ladle',
                'Lettuce',
                'LettuceSliced',
                'Mug',
                'Pan',
                'Plate',
                'Pot',
                'Potato',
                'PotatoSliced',
                'SoapBar',
                'Spatula',
                'Spoon',
                'Tomato',
                'TomatoSliced',
                'HandTowel'},
    'Cabinet': {'Book',
                'Bowl',
                'Box',
                'Candle',
                'CD',
                'Cloth',
                'Cup',
                'DishSponge',
                'Glassbottle',
                'Kettle',
                'Ladle',
                'Mug',
                'Newspaper',
                'Pan',
                'PepperShaker',
                'Plate',
                'Plunger',
                'Pot',
                'SaltShaker',
                'SoapBar',
                'SoapBottle',
                'SprayBottle',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Vase',
                'WateringCan',
                'WineBottle',
                'HandTowel'},
    'TableTop': {'AlarmClock',
                'Apple',
                'BaseballBat',
                'BasketBall',
                'Book',
                'Bowl',
                'Box',
                'Bread',
                'BreadSliced',
                'ButterKnife',
                'Candle',
                'CD',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'Cup',
                'DishSponge',
                'Glassbottle',
                'Egg',
                'Fork',
                'Kettle',
                'KeyChain',
                'Knife',
                'Ladle',
                'Laptop',
                'Lettuce',
                'LettuceSliced',
                'Mug',
                'Newspaper',
                'Pan',
                'PaperTowel',
                'Pen',
                'Pencil',
                'PepperShaker',
                'Plate',
                'Pot',
                'Potato',
                'PotatoSliced',
                'RemoteControl',
                'SaltShaker',
                'SoapBar',
                'SoapBottle',
                'Spatula',
                'Spoon',
                'SprayBottle',
                'Statue',
                'TennisRacket',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Tomato',
                'TomatoSliced',
                'Vase',
                'Watch',
                'WateringCan',
                'WineBottle',
                'HandTowel'},
    'CounterTop': {'AlarmClock',
                'Apple',
                'BaseballBat',
                'BasketBall',
                'Book',
                'Bowl',
                'Box',
                'Bread',
                'BreadSliced',
                'ButterKnife',
                'Candle',
                'CD',
                'CellPhone',
                'Cloth',
                'CreditCard',
                'Cup',
                'DishSponge',
                'Egg',
                'Glassbottle',
                'Fork',
                'Kettle',
                'KeyChain',
                'Knife',
                'Ladle',
                'Laptop',
                'Lettuce',
                'LettuceSliced',
                'Mug',
                'Newspaper',
                'Pan',
                'PaperTowel',
                'Pen',
                'Pencil',
                'PepperShaker',
                'Plate',
                'Pot',
                'Potato',
                'PotatoSliced',
                'RemoteControl',
                'SaltShaker',
                'SoapBar',
                'SoapBottle',
                'Spatula',
                'Spoon',
                'SprayBottle',
                'Statue',
                'TennisRacket',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Tomato',
                'TomatoSliced',
                'Vase',
                'Watch',
                'WateringCan',
                'WineBottle',
                'HandTowel'},
    'Shelf': {'AlarmClock',
            'Book',
            'Bowl',
            'Box',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'Cup',
            'DishSponge',
            'Glassbottle',
            'Kettle',
            'KeyChain',
            'Mug',
            'Newspaper',
            'PaperTowel',
            'Pen',
            'Pencil',
            'PepperShaker',
            'Plate',
            'Pot',
            'RemoteControl',
            'SaltShaker',
            'SoapBar',
            'SoapBottle',
            'SprayBottle',
            'Statue',
            'TissueBox',
            'ToiletPaper',
            'ToiletPaperRoll',
            'Vase',
            'Watch',
            'WateringCan',
            'WineBottle',
            'HandTowel'},
    'Drawer': {'Book',
            'ButterKnife',
            'Candle',
            'CD',
            'CellPhone',
            'Cloth',
            'CreditCard',
            'DishSponge',
            'Fork',
            'KeyChain',
            'Knife',
            'Ladle',
            'Newspaper',
            'Pen',
            'Pencil',
            'PepperShaker',
            'RemoteControl',
            'SaltShaker',
            'SoapBar',
            'SoapBottle',
            'Spatula',
            'Spoon',
            'SprayBottle',
            'TissueBox',
            'ToiletPaper',
            'ToiletPaperRoll',
            'Watch',
            'WateringCan',
            'HandTowel'},
    'GarbageCan': {'Apple',
                'Bread',
                'BreadSliced',
                'CD',
                'Cloth',
                'DishSponge',
                'Egg',
                'Lettuce',
                'LettuceSliced',
                'Newspaper',
                'PaperTowel',
                'Pen',
                'Pencil',
                'Potato',
                'PotatoSliced',
                'SoapBar',
                'SoapBottle',
                'SprayBottle',
                'TissueBox',
                'ToiletPaper',
                'ToiletPaperRoll',
                'Tomato',
                'TomatoSliced',
                'WineBottle',
                'HandTowel'},
    'Safe': {'CD',
            'CellPhone',
            'CreditCard',
            'KeyChain',
            'Statue',
            'Vase',
            'Watch'},
    'TVStand': {'TissueBox'},
    'Toaster': {'BreadSliced'},
}
VAL_RECEPTACLE_OBJECTS['DiningTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
VAL_RECEPTACLE_OBJECTS['CoffeeTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
VAL_RECEPTACLE_OBJECTS['SideTable'] = VAL_RECEPTACLE_OBJECTS['TableTop']
del VAL_RECEPTACLE_OBJECTS['TableTop']

MOVABLE_RECEPTACLES = [
    'Bowl',
    'Box',
    'Cup',
    'Mug',
    'Plate',
    'Pan',
    'Pot',
]

VAL_ACTION_OBJECTS = {
    'Heatable': {'Apple',
                 'AppleSliced',
                 'Bread',
                 'BreadSliced',
                 'Cup',
                 'Egg',
                 'Mug',
                 'Plate',
                 'Potato',
                 'PotatoSliced',  
                 'Tomato',
                 'TomatoSliced'},
    'Coolable': {'Apple',
                 'AppleSliced',
                 'Bowl',
                 'Bread',
                 'BreadSliced',
                 'Cup',
                 'Egg',
                 'Lettuce',
                 'LettuceSliced',
                 'Mug',
                 'Pan',
                 'Plate',
                 'Pot',
                 'Potato',
                 'PotatoSliced',
                 'Tomato',
                 'TomatoSliced',
                 'WineBottle'},
    'Cleanable': {'Apple',
                  'AppleSliced',
                  'Bowl',
                  'ButterKnife',
                  'Cloth',
                  'Cup',
                  'DishSponge',
                  'Egg',
                  'Fork',
                  'Kettle',
                  'Knife',
                  'Ladle',
                  'Lettuce',
                  'LettuceSliced',
                  'Mug',
                  'Pan',
                  'Plate',
                  'Pot',
                  'Potato',
                  'PotatoSliced',
                  'SoapBar',
                  'Spatula',
                  'Spoon',
                  'Tomato',
                  'TomatoSliced'},
    'Toggleable': {'DeskLamp',
                   'FloorLamp'},
    'Sliceable': {'Apple',
                  'Bread',
                  'Egg',
                  'Lettuce',
                  'Potato',
                  'Tomato'}
}

def check_unique_prediction(pddl_list):
    l = []
    for p in pddl_list:
        if p in l:
            continue
        else:
            l.append(p)
    return len(l) == 1

def preprocess(s):
    # remove escape sequence
    s = s.strip()
    if "\n" in s:
        s = s.split("\n")[0]
    elif "\b" in s:
        s = s.strip("\b")
    # CounterTop -> Counter Top
    if not any(_s.isupper() for _s in s):
        # lower
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

def get_task_type(s):
    dictionary = {
        'look_at_obj_in_light': 'look at object in light',
        'pick_and_place_with_movable_recep': 'pick and place with container',
        'pick_heat_then_place_in_recep': 'pick, heat then place',
        'pick_and_place_simple': 'pick and place',
        'pick_cool_then_place_in_recep': 'pick, cool then place',
        'pick_two_obj_and_place': 'pick two objects and place',
        'pick_clean_then_place_in_recep': 'pick, clean then place'}
    if s in dictionary.values():
        for k in dictionary:
            if dictionary[k] == s:
                return k
    return dictionary[s]

def generate_prompt(goal, sentences, plan_list, taskType=None):
    prompt = 'Make a plan to complete a given task\n'
    for i, example_task in enumerate(sentences):
        plan = plan_list[i]
        prompt += '%s :\n'%example_task
        for j, action in enumerate(plan):
            prompt += '%d. %s\n'%(j+1, action)
        prompt += '%dend'%(j+2)
        prompt += '\n\n'
    prompt += '%s :\n'%goal
    return prompt

def toAlfredID(action, word, recep = ''):
    if 'cd' in word:
        _word = 'CD'
    else:
        _word = word.title().replace(' ', '')
    recep = recep.title().replace(' ', '')
    if action == 'HeatObject':
        word_range = ['Apple','Bread','Cup','Egg','Mug','Plate','Potato','Tomato']
    elif action == 'CoolObject':
        word_range = ['Apple','Bowl','Bread','Cup','Egg','Lettuce','Mug','Pan','Plate','Pot','Potato','Tomato','WineBottle']
    elif action == 'CleanObject':
        word_range = ['Apple','Bowl','ButterKnife','Cloth','Cup','DishSponge','Egg','Fork','Kettle','Knife','Ladle','Lettuce','Mug','Pan','Plate','Pot','Potato','SoapBar','Spatula','Spoon','Tomato']
    elif action == 'ToggleObject':
        word_range = ['DeskLamp', 'FloorLamp']
    elif action == 'SliceObject':
        word_range = ['Apple','Bread','Egg','Lettuce','Potato','Tomato']
    elif action == 'PutObject-o':
        if recep == '' or recep not in VAL_RECEPTACLE_OBJECTS.keys():
            word_range = OBJECTS
        else:
            word_range = VAL_RECEPTACLE_OBJECTS[recep]
    elif action == 'PutObject-r':
        word_range = list(VAL_RECEPTACLE_OBJECTS.keys())
    else:
        word_range = OBJECTS
    if _word in word_range:
        return _word.lower()
    return closest_object_roberta(word, set(word_range))[0].lower()

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translation_lm = SentenceTransformer('stsb-roberta-large').to(device)
    example_task_embedding = translation_lm.encode(oList, batch_size=512, convert_to_tensor=True, device=device)
    query_embedding = translation_lm.encode(preprocess(key), convert_to_tensor=True, device=device)
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, example_task_embedding)[0].detach().cpu().numpy()
    idx = np.argsort(cos_scores)[-1]
    most_similar_object, matching_score = set[idx], cos_scores[idx]
    # print('Word %s replaced with %s (Matching score: %.4f)'%(key, most_similar_object, matching_score*100))
    return most_similar_object, matching_score

def toAlfredPlan(sentence, available_actions):
    plan = sentence.split('\n')
    action_seq = []
    available_tasks = set(available_actions.keys())
    # match sugoal generated by GPT into ALFRED available subgoals
    for action in plan:
        # Remove empty string
        if action.strip() == '':
            continue
        action = re.sub(r"[0-9]", "", action).strip('. ').lower()
        action_dict = {}
        most_similar_task, _ = closest_object_roberta(action.lower(), available_tasks)
        if available_actions[most_similar_task] is not None:
            action_seq.append(available_actions[most_similar_task])
    # Remove repititive subgoals
    temp = []
    for i, action in enumerate(action_seq):
        if i != 0 and action == action_seq[i-1]:
            continue
        temp.append(action)
    action_seq = temp

    return action_seq

def get_available_actions():
    actions = {"end": None}
    for obj in OBJECTS:
        if 'sliced' in obj.lower():
            continue
        if (obj not in VAL_RECEPTACLE_OBJECTS.keys() or obj in MOVABLE_RECEPTACLES):
            actions["pick %s up"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "PickupObject", "args": [obj]}
        if obj in VAL_ACTION_OBJECTS['Heatable']:
            actions["heat %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "HeatObject", "args": [obj]}
            actions["cook %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "HeatObject", "args": [obj]}
        if obj in VAL_ACTION_OBJECTS['Toggleable']:
            actions["turn %s on"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "ToggleObject", "args": [obj]}
            actions["examine it with %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "ToggleObject", "args": [obj]}
        if obj in VAL_ACTION_OBJECTS['Cleanable']:
            actions["clean %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "CleanObject", "args": [obj]}
            actions["wash %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "CleanObject", "args": [obj]}
        if obj in VAL_ACTION_OBJECTS['Coolable']:
            actions["cool %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "CoolObject", "args": [obj]}
            actions["chill %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "CoolObject", "args": [obj]}
        if obj in VAL_ACTION_OBJECTS['Sliceable']:
            actions["slice %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "SliceObject", "args": [obj]}
            actions["cut %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower())] = {"action": "SliceObject", "args": [obj]}
    for recep, obj_set in VAL_RECEPTACLE_OBJECTS.items():
        for obj in obj_set:
            if 'sliced' in obj.lower():
                continue
            actions["put %s on %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower(), re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", recep).lower())] = {"action": "PutObject", "args": [obj, recep]}
            actions["place %s on %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower(), re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", recep).lower())] = {"action": "PutObject", "args": [obj, recep]}
            actions["drop %s on %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower(), re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", recep).lower())] = {"action": "PutObject", "args": [obj, recep]}
            actions["arrange %s on %s"%(re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", obj).lower(), re.sub(r"((?<=[a-z])[A-Z])|([A-Z](?=[A-Z][a-z]))", r" \1", recep).lower())] = {"action": "PutObject", "args": [obj, recep]}
    return actions

def response2plan(sentence):
    plan = sentence.split('\n')
    action_seq = []
    for action in plan:
        action = re.sub(r"[0-9]", "", action).strip('. ').lower()
        action_dict = {}
        if 'go' in action:
            action_dict['action'] = 'GotoLocation'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"go| to ", "", action))]
        elif 'pick' in action:
            action_dict['action'] = 'PickupObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"pick| up", "", action))]
        elif 'put' in action:
            action_dict['action'] = 'PutObject'
            action = re.sub(r"put| on | in ", '-', action)
            action = action.replace(' down', '')
            if len(action.split('-')) < 3:
                action_dict['args'] = [toAlfredID(action_dict['action']+'-o', action.split('-')[1]), '']
            else:
                recep = toAlfredID(action_dict['action']+'-r', action.split('-')[2])
                action_dict['args'] = [toAlfredID(action_dict['action']+'-o', action.split('-')[1], recep=recep), recep]
        elif 'place' in action:
            action_dict['action'] = 'PutObject'
            action = re.sub(r"place | on | in ", '-', action)
            action = action.replace(' down', '')
            if len(action.split('-')) < 3:
                print(action)
                action_dict['args'] = [toAlfredID(action_dict['action']+'-o', action.split('-')[1]), '']
            else:
                recep = toAlfredID(action_dict['action']+'-r', action.split('-')[2])
                action_dict['args'] = [toAlfredID(action_dict['action']+'-o', action.split('-')[1], recep=recep), recep]
        elif 'drop' in action:
            action_dict['action'] = 'PutObject'
            action = re.sub(r"drop | on | in ", '-', action)
            action = action.replace(' down', '')
            if len(action.split('-')) < 3:
                action_dict['args'] = [toAlfredID(action_dict['action']+'-o', action.split('-')[1]), '']
            else:
                recep = toAlfredID(action_dict['action']+'-r', action.split('-')[2])
                action_dict['args'] = [toAlfredID(action_dict['action']+'-o', action.split('-')[1], recep=recep), recep]
        elif 'arrange' in action:
            action_dict['action'] = 'PutObject'
            action = re.sub(r"arrange| on | in ", '-', action)
            action = action.replace(' down', '')
            if len(action.split('-')) < 3:
                action_dict['args'] = [toAlfredID(action_dict['action'], action.split('-')[1]), '']
            else:
                recep = toAlfredID(action_dict['action']+'-r', action.split('-')[2])
                action_dict['args'] = [toAlfredID(action_dict['action'], action.split('-')[1], recep=recep), recep]
        elif 'take' in action:
            action_dict['action'] = 'PutObject'
            action = re.sub(r"take | on | in | to ", '-', action)
            action = action.replace(' down', '')
            if len(action.split('-')) < 3:
                action_dict['args'] = [toAlfredID(action_dict['action'], action.split('-')[1]), '']
            else:
                recep = toAlfredID(action_dict['action']+'-r', action.split('-')[2])
                action_dict['args'] = [toAlfredID(action_dict['action'], action.split('-')[1], recep=recep), recep]
        elif 'turn' in action:
            action_dict['action'] = 'ToggleObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"turn| on", "", action))]
        elif 'examine' in action:
            action_dict['action'] = 'ToggleObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"examine| on", "", action))]
        elif 'clean' in action:
            action_dict['action'] = 'CleanObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"clean", "", action))]
        elif 'wash' in action:
            action_dict['action'] = 'CleanObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"wash", "", action))]
        elif 'cool' in action:
            action_dict['action'] = 'CoolObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"cool", "", action))]
        elif 'chill' in action:
            action_dict['action'] = 'CoolObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"chill", "", action))]
        elif 'heat' in action:
            action_dict['action'] = 'HeatObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"heat", "", action))]
        elif 'cook' in action:
            action_dict['action'] = 'HeatObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"cook", "", action))]
        elif 'slice' in action:
            action_dict['action'] = 'SliceObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"slice", "", action))]
        elif 'cut' in action:
            action_dict['action'] = 'SliceObject'
            action_dict['args'] = [toAlfredID(action_dict['action'], re.sub(r"cut ", "", action))]
        else:
            print('Not Transferred to action dict \"%s\"'%action)
            continue
        action_seq.append(action_dict)
    return action_seq

def main(args):
    # load (plan_match: {goal: plan list}) and (sentence_match: {goal: found}) Data
    plan_match_file = 'result/alfred/roberta/valid_%s-plan@20.json'%args.split
    sentence_match_file= 'result/alfred/roberta/valid_%s-sentence@20.json'%args.split
    save_file = 'result/alfred/prompt/valid_%s-plan@20(Line)-v2.json'%args.split
    if args.save_prompts:
        prompt_file = 'result/alfred/prompt/valid_%s-planPrompt@20(Line)-v2.json'%args.split

    if not os.path.exists(os.path.split(save_file)[0]):
        os.makedirs(os.path.split(save_file)[0], exist_ok=True)

    with open(plan_match_file, 'r') as f:
        plan_match = json.load(f)
        temp = {}
        for goal in plan_match:
            temp[goal.split('[SEP]')[0]] = plan_match[goal]
        plan_match = temp
        if args.debug:
            # choose number to sample
            sample = random.sample(plan_match.keys(), 40)
            vals = [plan_match[k] for k in sample]
            plan_match = dict(zip(sample, vals))
    with open(sentence_match_file, 'r') as f:
        sentence_match = json.load(f)
        temp = {}
        for goal in sentence_match:
            temp[goal.split('[SEP]')[0]] = [s.split('[SEP]')[0] for s in sentence_match[goal]]
        sentence_match = temp
        if args.debug:
            temp = {}
            for goal in plan_match:
                temp[goal] = sentence_match[goal]
            sentence_match = temp
    
    # Resume
    if os.path.exists(save_file):    
        with open(save_file, 'r') as f:
            gpt_plan_match = json.load(f)
    else:
        gpt_plan_match = {}
    _gpt_plan_match = {}
    if args.save_prompts:
        if os.path.exists(prompt_file):    
            with open(prompt_file, 'r') as f:
                prompts = json.load(f)
        else:
            prompts = []
        _prompts = []

    available_actions = get_available_actions()

    # Generate pddl parameter
    for goal, plan_list in tqdm(plan_match.items()):
        if goal in gpt_plan_match:
            # this goal already matched
            continue

        sentences = sentence_match[goal] 
        prompt = generate_prompt(goal, sentences, plan_list)
        given_prompt = prompt
        replace = []

        action_index = 1
        # Get gpt3 response
        while True:
            try:
                response = openai.Completion.create(
                    model = 'text-davinci-003',
                    prompt = prompt,
                    top_p = 1,
                    max_tokens = 500,
                    stop = '\n'
                )
                text = response.choices[0].text
                if text.strip() == '':
                    continue
                if 'end' in text:
                    break
                subgoal, _ = closest_object_roberta(text, set(available_actions.keys()))
                replace.append((text, subgoal))
                prompt = prompt + '%d. %s\n'%(action_index, subgoal)
                action_index += 1
            except openai.error.ServiceUnavailableError:
                time.sleep(0.3)

        if args.save_prompts:
            _prompts.append((prompt+response.choices[0].text, ' -> '.join([r[1] for r in replace])))

        # Process result string to dict
        _gpt_plan_match[goal] = toAlfredPlan(prompt.replace(given_prompt, ''), available_actions)
  
        # Save pddl match
        if len(list(_gpt_plan_match.keys())) > 5:
            if os.path.exists(save_file):    
                with open(save_file, 'r') as f:
                    gpt_plan_match = json.load(f)
            else:
                gpt_plan_match = {}
            gpt_plan_match.update(_gpt_plan_match)
            with open(save_file, 'w') as f:
                json.dump(gpt_plan_match, f, indent=4)        
            _gpt_plan_match = {}
            if args.save_prompts:
                if os.path.exists(prompt_file):    
                    with open(prompt_file, 'r') as f:
                        prompts = json.load(f)
                else:
                    prompts = []
                prompts.extend(_prompts)
                with open(prompt_file, 'w') as f:
                    json.dump(prompts, f, indent=4)        
                _prompts = []
            
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            gpt_plan_match = json.load(f)
    else:
        gpt_plan_match = {}
    _gpt_plan_match.update(gpt_plan_match)
    with open(save_file, 'w') as f:
        json.dump(_gpt_plan_match, f, indent=4)       
    if args.save_prompts:
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompts = json.load(f)
        else:
            prompts = {}
        prompts.extend(_prompts)
        with open(prompt_file, 'w') as f:
            json.dump(prompts, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_prompts', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--split', choices=['seen', 'unseen'], required=True)
    args = parser.parse_args()
    main(args)