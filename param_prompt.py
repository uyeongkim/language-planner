import json
import random 
import re
import openai
import time
import argparse
import os
import numpy as np
from tqdm import tqdm

# Yonsei api key
openai.api_key = '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'
openai.organization = 'org-azdthpxrguDHQc2ujvxf4hTZ'

# yuyeong personal api key
# openai.api_key = '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'
# openai.organization = 'org-pLk6QBv25E0v68rzU9KfJXiu'

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

def generate_prompt(goal, sentences, pddl_list, taskType=None):
    if taskType == None:
        prompt = "Fill in the table below.\n"
        prompt += "goal sentence | task type | container to put target in | target object | should slice the target object | place to put target | lamp to turn on\n"
        for i, s in enumerate(sentences):
            pddl = pddl_list[i]
            m = pddl['mrecep_target'] if pddl['mrecep_target'] != "" else "None"
            p = pddl['parent_target'] if pddl['parent_target'] != "" else "None"
            t = pddl['toggle_target'] if pddl['toggle_target'] != "" else "None"
            task_type = get_task_type(pddl['task_type'])
            prompt += '{} | {} | {} | {} | {} | {} | {}\n'.format(s, task_type, m, pddl['object_target'], pddl['object_sliced'], p, t)
        prompt += preprocess(goal)
        return prompt

    # Ablation study for gt taskType
    pred_args = {
        "pick_and_place_simple": [False, True, True, True, False],
        "pick_cool_then_place_in_recep": [False, True, True, True, False],
        "pick_and_place_with_movable_recep": [True, True, True, True, False],
        "pick_two_obj_and_place": [False, True, True, True, False],
        "pick_heat_then_place_in_recep": [False, True, True, True, False],
        "look_at_obj_in_light": [False, True, True, False, True],
        "pick_clean_then_place_in_recep": [False, True, True, True, False],
    }
    head = np.array(["container to put target in", "target object", \
        "should slice the target object", "place to put target", "lamp to turn on"])[np.array(pred_args[taskType])]
    
    prompt = "Fill in the table below.\n"
    prompt += "goal sentence | "+" | ".join(list(head))+"\n"
    for i, s in enumerate(sentences):
        pddl = pddl_list[i]
        m = pddl['mrecep_target'] if pddl['mrecep_target'] != "" else "None"
        p = pddl['parent_target'] if pddl['parent_target'] != "" else "None"
        t = pddl['toggle_target'] if pddl['toggle_target'] != "" else "None"
        args_list = np.array([m, pddl['object_target'], pddl['object_sliced'], p, t])[np.array(pred_args[taskType])]
        prompt += "{} | ".format(s)+" | ".join(list(args_list))+"\n"
    prompt += preprocess(goal)
    return prompt

def response2param(response, taskType=None):
    if taskType == None:
        result = response.choices[0].text
        ks = ['task_type', 'mrecep_target', 'object_target', 'object_sliced', 'parent_target', 'toggle_target']
        l = result.split('|')[-6:]
        try:
            vs = [get_task_type(l[0].strip())]
        except:
            vs = [l[0]]
        for _l in l[1:]:
            if ',' in _l and not _l.startswith(','):
                _l = _l.split(',')[0]
            if 'false' in _l.lower():
                vs.append(False)
                continue
            if 'true' in _l.lower():
                vs.append(True)
                continue
            if 'none' in _l.lower():
                vs.append("")
                continue
            vs.append(_l.strip())
        result = dict(zip(ks, vs))
        return result
    result = response.choices[0].text
    pred_args = {
        "pick_and_place_simple": [False, True, True, True, False],
        "pick_cool_then_place_in_recep": [False, True, True, True, False],
        "pick_and_place_with_movable_recep": [True, True, True, True, False],
        "pick_two_obj_and_place": [False, True, True, True, False],
        "pick_heat_then_place_in_recep": [False, True, True, True, False],
        "look_at_obj_in_light": [False, True, True, False, True],
        "pick_clean_then_place_in_recep": [False, True, True, True, False],
    }
    ks = np.array(['mrecep_target', 'object_target', 'object_sliced', 'parent_target', 'toggle_target'])[np.array(pred_args[taskType])]
    l = result.split('|')[-sum(pred_args[taskType]):]
    vs = []
    for _l in l:
        if ',' in _l and not _l.startswith(','):
            _l = _l.split(',')[0]
        if 'false' in _l.lower():
            vs.append(False)
            continue
        if 'true' in _l.lower():
            vs.append(True)
            continue
        if 'none' in _l.lower():
            vs.append("")
            continue
        vs.append(_l.strip())
    result = dict(zip(ks, vs))
    for k in ['mrecep_target', 'object_target', 'object_sliced', 'parent_target', 'toggle_target']:
        if k not in result:
            result[k] = ""
    return result

def main(args):
    # load (pddl_match: {goal: pddl}) and (sentence_match: {goal: found}) Data
    if args.template:
        pddl_match_file = 'result/fullTemplate/stsb-roberta-large-pddl_match@%d.json'%args.k
        sentence_match_file= 'result/fullTemplate/stsb-roberta-large-sentence_match@%d.json'%args.k
        if args.gtType:
            save_file = 'result/fullTemplate/gpt3/gtType/top%d.json'%args.k
            prompt_file = 'result/fullTemplate/gpt3/gtType/top%d-prompts.json'%args.k
        else:
            save_file = 'result/fullTemplate/gpt3/top%d.json'%args.k
            prompt_file = 'result/fullTemplate/gpt3/top%d-prompts.json'%args.k
    else:
        pddl_match_file = 'result/alfred/appended/roberta/%s-argument@%d.json'%(args.split, args.k)
        sentence_match_file= 'result/alfred/appended/roberta/%s-sentence@%d.json'%(args.split, args.k)
        if args.gtType:
            save_file = 'result/fullTemplate/gpt3/gtType/top%d.json'%args.k
            prompt_file = 'result/fullTemplate/gpt3/gtType/top%d-prompts.json'%args.k
        else:
            save_file = 'result/alfred/appended/prompt/%s-argument@%d.json'%(args.split, args.k)
            prompt_file = 'result/alfred+/apended/prompt/%s-prompt@%d.json'%(args.split, args.k)

    if not os.path.exists(os.path.split(save_file)[0]):
        os.makedirs(os.path.split(save_file)[0], exist_ok=True)
    with open(pddl_match_file, 'r') as f:
        pddl_match = json.load(f)
        temp = {}
        for goal in pddl_match:
            temp[goal.split('[SEP]')[0]] = pddl_match[goal]
        pddl_match = temp
    with open(sentence_match_file, 'r') as f:
        sentence_match = json.load(f)
        temp = {}
        for goal in sentence_match:
            temp[goal.split('[SEP]')[0]] = [s.split('[SEP]')[0] for s in sentence_match[goal]]
        sentence_match = temp
    
    # Resume
    if os.path.exists(save_file):    
        with open(save_file, 'r') as f:
            gpt_pddl_match = json.load(f)
    else:
        gpt_pddl_match = {}
    _gpt_pddl_match = {}
    if args.save_prompts:
        if os.path.exists(prompt_file):    
            with open(prompt_file, 'r') as f:
                prompts = json.load(f)
        else:
            prompts = []
        _prompts = []

    if args.gtType:
        with open('data/taskType.json', 'r') as f:
            taskType = json.load(f)
        temp = dict()
        for k in taskType:
            temp[k.strip()] = taskType[k]
        taskType = temp
    if args.debug:
        _pddl_match = {}
        key_sample = random.sample(pddl_match.keys(), 5)
        for key in key_sample:
            _pddl_match[key] = pddl_match[key]
        pddl_match = _pddl_match
    # Generate pddl parameter
    for goal, pddl_list in tqdm(pddl_match.items()):
        if goal in gpt_pddl_match and len(list(gpt_pddl_match[goal].keys())) == 6:
            # this goal already matched
            continue
        if args.gtType and goal not in taskType:
            print(goal)
            raise Exception("not in taskType.json")

        sentences = sentence_match[goal]
        if args.gtType:
            prompt = generate_prompt(goal, sentences, pddl_list, taskType[goal])
        else:    
            prompt = generate_prompt(goal, sentences, pddl_list)

        # Get gpt3 response
        response = openai.Completion.create(
            model = 'text-davinci-002',
            prompt = prompt,
            temperature = 0.1,
            max_tokens = 300,
            stop = ['\n']
        )

        if args.save_prompts:
            _prompts.append(prompt+response.choices[0].text)

        # Process result string to dict
        if args.gtType:
            _gpt_pddl_match[goal] = response2param(response, taskType[goal])
        else:
            _gpt_pddl_match[goal] = response2param(response)
        print('Added')
        time.sleep(1.2)
  
        # Save pddl match
        if len(list(_gpt_pddl_match.keys())) > 5:
            print('\n---Saving---')
            if os.path.exists(save_file):    
                with open(save_file, 'r') as f:
                    gpt_pddl_match = json.load(f)
            else:
                gpt_pddl_match = {}
            gpt_pddl_match.update(_gpt_pddl_match)
            with open(save_file, 'w') as f:
                json.dump(gpt_pddl_match, f, indent=4)        
            _gpt_pddl_match = {}
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
            gpt_pddl_match = json.load(f)
    else:
        gpt_pddl_match = {}
    _gpt_pddl_match.update(gpt_pddl_match)
    with open(save_file, 'w') as f:
        json.dump(_gpt_pddl_match, f, indent=4)       
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
    parser.add_argument('--k', type=int, help='top k')
    parser.add_argument('--save_prompts', action='store_true')
    parser.add_argument('--template', action='store_true')
    parser.add_argument('--gtType', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--split', type=str)
    args = parser.parse_args()
    main(args)