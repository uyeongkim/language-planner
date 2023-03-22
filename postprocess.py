import argparse
import json
import torch
import sys
import os
import numpy as np
from sentence_transformers import SentenceTransformer
# import torchtext
from sentence_transformers import util as st_utils
sys.path.append('../alfred')
sys.path.append('../alfred/gen')
from gen import constants
import re
from tqdm import tqdm

# all pickable objsb -- sliced excluded
OBJECTS_SET = constants.OBJECTS_SET
# for put action {parent:obj}
RECEPTACLE_MATCH = constants.VAL_RECEPTACLE_OBJECTS
ACTION_MATCH = constants.VAL_ACTION_OBJECTS
MRECEP_SET = constants.MOVABLE_RECEPTACLES_SET
RECEP_SET = constants.RECEPTACLES
SINGULAR = constants.OBJECTS_SINGULAR
PLURAL = constants.OBJECTS_PLURAL

def get_admissible_set(category):
    if category == "mrecep_target":
        return MRECEP_SET | {""}
    if category == "object_target":
        return OBJECTS_SET-(RECEP_SET-MRECEP_SET)
    if category == "object_sliced":
        return {'True', 'False'}
    if category == "parent_target":
        return RECEP_SET | {""}
    if category == "toggle_target":
        return {"DeskLamp", "FloorLamp", ""}
    raise Exception("Category name is not in pddl parameter")

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

def closest_object_cosine(key, set):
    if key == '':
        return np.random.choice(list(set), 1)[0], 0
    if type(key) == type(True):
        return key, 0
    glove = torchtext.vocab.GloVe(name="6B", dim=100)
    if key == 'Laddle':
        key = 'ladle'
    score = 0
    word = key
    for w in set:
        if '' == w:
            continue
        _w = preprocess(w)
        if ' ' in _w:
            wRpr = sum([glove[r] for r in _w.split()])
        elif _w == 'glassbottle':
            wRpr == glove['glass'] + glove['bottle']
        else:
            wRpr = glove[_w]
        _word = preprocess(word)
        if ' ' in _word:
            wordRpr = sum([glove[r] for r in _word.split()])
        elif _word == 'glassvases':
            wordRpr = glove['glass'] + glove['vases']
        elif _word == 'cd\'s':
            wordRpr = glove['cds']
        elif _word == 'glassjug':
            wordRpr = glove['glass'] + glove['jug']
        elif _word == 'glassbottle':
            wordRpr = glove['glass'] + glove['bottle']
        else:
            wordRpr = glove[_word]
        if not wRpr.any() or not wordRpr.any():
            print(word)
            print(w)
            raise Exception('word not in dictionary')
        _s = torch.cosine_similarity(wordRpr.unsqueeze(0), wRpr.unsqueeze(0))
        if _s > score:
            score = _s
            word = w
    if word == key:
        # All objects in admissible set are not similar
        word = np.random.choice(list(set), 1)[0]
    return word, score

def closest_object_l2(key, set):
    if key == '':
        return np.random.choice(list(set), 1)[0], 0
    glove = torchtext.vocab.GloVe(name="6B", dim=100)
    if type(key) == type(True):
        return key, 0
    if key == 'Laddle':
        key = 'ladle'
    score = float('inf')
    word = key
    for w in set:
        if '' == w:
            continue
        _w = preprocess(w)
        if ' ' in _w:
            wRpr = sum([glove[r] for r in _w.split()])
        elif _w == 'glassbottle':
            wRpr == glove['glass'] + glove['bottle']
        else:
            wRpr = glove[_w]
        _word = preprocess(word)
        if ' ' in _word:
            wordRpr = sum([glove[r] for r in _word.split()])
        elif _word == 'glassvases':
            wordRpr = glove['glass'] + glove['vases']
        elif _word == 'cd\'s':
            wordRpr = glove['cds']
        elif _word == 'glassjug':
            wordRpr = glove['glass'] + glove['jug']
        elif _word == 'glassbottle':
            wordRpr = glove['glass'] + glove['bottle']
        else:
            wordRpr = glove[_word]
        if not wRpr.any() or not wordRpr.any():
            print(word)
            print(w)
            raise Exception('word not in dictionary')
        _s = torch.norm(wordRpr-wRpr)
        if _s < score:
            score = _s
            word = w
    if word == key:
        # All objects in admissible set are not similar
        word = np.random.choice(list(set), 1)[0]
    return word, score

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
    return most_similar_object, matching_score

def save_pddl_match(pddl_match, file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.split(file_path)[0], exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(pddl_match, f, indent=4)

def load_pddl_match(file_path):
    with open(file_path, 'r') as f:
        pddl_match = json.load(f)
    return pddl_match

def main(args):
    with open(args.inFile, 'r') as f:
        pddl_match = json.load(f)
    try:
        new_pddl_match = load_pddl_match(args.outFile)
    except:
        new_pddl_match = dict()
    count = 0
    for goal, pddl in tqdm(pddl_match.items()):
        count += 1
        params = []
        if goal in new_pddl_match:
            continue
        for category, obj in pddl.items():
            if category == 'object_sliced' and type(obj) == bool:
                params.append(obj)
                continue
            oSet = get_admissible_set(category)
            if obj in oSet:
                params.append(obj)
                continue
            if type(obj) == type(True):
                obj = 'True' if obj else 'False'
            if args.metric == 'l2':
                matched_obj, score = closest_object_l2(obj, oSet)    
            elif args.metric == 'cosine':
                matched_obj, score = closest_object_cosine(obj, oSet)
            elif args.metric == 'roberta':
                matched_obj, score = closest_object_roberta(obj, oSet)
            if category == 'object_sliced':
                matched_obj = matched_obj == 'True'
            params.append(matched_obj)
        new_pddl_match[goal] = dict(zip(pddl.keys(), params))
        if count == 5:
            try:
                save_pddl_match(new_pddl_match, args.outFile)
            except:
                save_pddl_match(new_pddl_match, args.inFile.split('.')[0]+"-%s_match.json"%args.metric)
            count = 0
    try:
        save_pddl_match(new_pddl_match, args.outFile)
    except:
        save_pddl_match(new_pddl_match, args.inFile.split('.')[0]+"-%s_match.json"%args.metric)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inFile", type=str)
    parser.add_argument("--outFile", type=str, default="")
    parser.add_argument('--metric', type=str, default='roberta')
    args = parser.parse_args()
    main(args)
