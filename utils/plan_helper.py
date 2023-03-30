"""
Useful functions dealing with plans
"""
import re
import json
from multiprocessing import Pool, Value, Queue, Process
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import openai
from utils import config

openai.api_key = config.OPENAI['api_key']
openai.organization = config.OPENAI['organization']
def count_plans(plans) -> int:
    """
    Args:
        plans: list of plans
    Returns:
        # of different plans
    """
    repr = []
    for plan in plans:
        if plan not in repr:
            repr.append(plan)
    return len(repr)

def toString(plan) -> str:
    """
    Args:
        plan: list of action strings
    Returns:
        "1. action\n 2.action ..."
    """
    desc = ""
    for i, action in enumerate(plan):
        desc += '%d. %s\n'%(i+1, action)
    return desc

# def is_same_plan(plan1, plan2) -> bool:
#     """
#     Args:
#         plan1, plan2: list of action dicts
#         ["action": "", "args": []]
#     Returns:
#         True if two plans are the same
#     """
#     if len(plan1) != len(plan2):
#         return False
#     for i in range(len(plan1)):
#         if plan1[i]["action"].lower() != plan2[i]["action"].lower():
#             return False
#         if len(plan1[i]["args"]) != len(plan2[i]["args"]):
#             return False
#         for j in range(len(plan1[i]["args"])):
#             if plan1[i]["args"][j].lower() != plan2[i]["args"][j].lower():
#                 return False
#     return True

def remove_duplicate(plans):
    result = []
    for plan in plans:
        if plan not in result:
            result.append(plan)
    return result

def closest_object_roberta(key, corpus):
    corpus = list(set(corpus))
    oList = [preprocess_goal(o) for o in corpus]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    translation_lm = SentenceTransformer('stsb-roberta-large').to(device)
    example_task_embedding = translation_lm.encode(oList, batch_size=512, convert_to_tensor=True, device=device)
    query_embedding = translation_lm.encode(preprocess_goal(key), convert_to_tensor=True, device=device)
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, example_task_embedding)[0].detach().cpu().numpy()
    idx = np.argsort(cos_scores)[-1]
    most_similar_object, matching_score = corpus[idx], cos_scores[idx]
    return most_similar_object, matching_score

def gptResponseToAlfredPlan(sentence, available_actions):
    plan = sentence.split('\n')
    action_seq = []; cos_score = []
    available_tasks = set(available_actions.keys())
    # match sugoal generated by GPT into ALFRED available subgoals
    for action in plan:
        # Remove empty string
        if action.strip() == '':
            continue
        if ':' in action and not any(temp.isdigit() for temp in action):
            break
        action = re.sub(r"[0-9]", "", action).strip('. ').lower()
        most_similar_task, score = closest_object_roberta(action.lower(), available_tasks)
        action_seq.append(available_actions[most_similar_task])
        cos_score.append(float(score))
    # Remove repititive subgoals
    temp = []
    for i, action in enumerate(action_seq):
        if i != 0 and action == action_seq[i-1]:
            continue
        temp.append(action)
    action_seq = temp
    return action_seq, cos_score

def preprocess_goal(s):
    # remove escape sequence
    s = s.strip()
    if "\n" in s:
        s = s.split("\n")[0]
    if "\b" in s:
        s = s.strip("\b")
    s = s.strip()
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
            lteft = ''
    if len(ls) == 0:
        ls = left.split()
    # lower
    return ' '.join(ls).lower().replace('.', '')

def make_triplet_plan(traj_data) -> list:
    """
    Args:
        traj_data(dict):
            'plan': , 'task_type': , ...
    Returns:
        plan(list):
            [ [action, object, recep], ... ]
    """
    def find_recep(action_list, obj):
        # [A, O, R]
        for a in action_list:
            if a[0] == 'PutObject' and a[1] == obj:
                return a[2]
            if a[0] == 'SliceObject' and a[1] == obj:
                return a[2]
        return ''
    from data.alfred_data import constants
    alfred_obj_lower_to_id = dict(zip(constants.OBJECTS_SINGULAR ,constants.OBJECTS))

    plan = []
    for i, action in enumerate(traj_data['plan']['high_pddl']):
        if action['discrete_action']['action'] in ['GotoLocation', 'NoOp']:
            continue
        try:
            plan.append([action['discrete_action']['action'], \
                action['planner_action']['coordinateObjectId'][0], action['planner_action']['coordinateReceptacleObjectId'][0]])
        except KeyError as e:
            if e.args[0] == 'coordinateReceptacleObjectId':
                if action['discrete_action']['action'] == 'SliceObject':
                    # if later put
                    replaced = False
                    for _action in traj_data['plan']['high_pddl'][i:]:
                        if _action['discrete_action'] == {"action":"PickupObject", "args":[action['planner_action']['coordinateObjectId'][0].lower()]}:
                            replaced = True
                            try:
                                plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], \
                                    _action['planner_action']['coordinateReceptacleObjectId'][0]])
                            except:
                                plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], ''])
                    if not replaced:
                        # if putted bf: putted already
                        if find_recep(plan, action['planner_action']['coordinateObjectId'][0]) == '':
                            if traj_data["task_type"] == "look_obj_in_light":
                                if traj_data['plan']['high_pddl']['discrete_action']['action'] != 'GotoLocation':
                                    plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], \
                                        traj_data['pddl_params']['toggle_target']])
                                else:
                                    plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], ''])
                            elif traj_data["task_type"] == "pick_and_place_with_movable_recep":
                                plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], \
                                    traj_data['pddl_params']['mrecep_target']])
                            else:
                                plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], \
                                    traj_data['pddl_params']['parent_target']])
                        else:    
                            plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], \
                                find_recep(plan, action['planner_action']['coordinateObjectId'][0])])
                elif traj_data['task_type'] != 'pick_and_place_with_movable_recep':
                    plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], ''])
                else:
                    if find_recep(plan, action['planner_action']['coordinateObjectId'][0]) == '' and i != 0 \
                        and traj_data['plan']['high_pddl'][i-1]['discrete_action']['action'] != 'GotoLocation':
                        plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], \
                            plan[-1][-1]])
                    else:    
                        plan.append([action['discrete_action']['action'], action['planner_action']['coordinateObjectId'][0], \
                            find_recep(plan, action['planner_action']['coordinateObjectId'][0])])
            elif e.args[0] == 'coordinateObjectId':
                if action['discrete_action']['args'][0] != '':
                    plan.append([action['discrete_action']['action'], \
                        alfred_obj_lower_to_id[action['discrete_action']['args'][0]], action['planner_action']['coordinateReceptacleObjectId'][0]])
                elif plan[-1][0] == 'SliceObject':
                    plan.append([action['discrete_action']['action'], \
                        plan[-2][1], action['planner_action']['coordinateReceptacleObjectId'][0]])
                else:
                    plan.append([action['discrete_action']['action'], \
                        plan[-1][1], action['planner_action']['coordinateReceptacleObjectId'][0]])
            else:
                raise e
    return plan

def write_plan_log(pred_dict, save_file):
    """
    Args:
        pred_dict(dict):
            { goal: {"plan": [], ...}}
        (Optional)
            prompt_list(list): lists of prompts corresponding to pred_dict
    """
    val_seen_path = 'data/alfred_data/json_2.1.0/valid_seen'

    count = 0
    fail = 0
    ksuccess = 0
    log = ''
    for ep in os.listdir(val_seen_path):
        for trial in os.listdir(os.path.join(val_seen_path, ep)):
            traj_data = json.load(open(os.path.join(val_seen_path, ep, trial, 'traj_data.json')))
            gt = [action_block['discrete_action'] for action_block in traj_data['plan']['high_pddl'] if action_block['discrete_action']['action'] not in ['NoOp', 'GotoLocation']]
            anns = [ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]
            for goal in anns:
                try:
                    preds = pred_dict[goal]['plan']
                    count += 1
                except:
                    continue
                ksuc = False
                suc = True
                for plan in preds:
                    success = (len(gt) == len(plan))
                    for i, action in enumerate(gt):
                        if not success:
                            break
                        if action['action'] != plan[i]['action'] or [a.lower() for a in action['args']] != [a.lower() for a in plan[i]['args']]:
                            success = False
                    if not success:
                        suc = False
                        log += '\nTask: %s\n'%goal
                        log += '%-25s %-25s\n'%('Pred', 'GT')
                        row = max(len(gt), len(plan))
                        for r in range(row):
                            p_action = plan[r]['action'].replace('Object', '') if len(plan) > r else ''
                            p_args = ' '.join(plan[r]['args'])  if len(plan) > r else ''
                            g_action = gt[r]['action'].replace('Object', '') if len(gt) > r else ''
                            g_arg = ' '.join(gt[r]['args'])  if len(gt) > r else ''
                            log += '%-25s %-25s\n'%(p_action+' '+p_args, g_action+' '+g_arg)
                    else:
                        ksuc = True
                if not suc:
                    fail += 1
                if ksuc:
                    ksuccess += 1

    log += '@K Acc: %d / %d (%.2f)\n'%(ksuccess, count, ksuccess/count*100)
    log += 'Acc: %d / %d (%.2f)\n'%(count-fail, count, (count-fail)/count*100)
    with open(save_file, 'w') as f:
        f.write(log)

def compare_plan(plan1, plan2):
    """
    plan1 is baseline plan
    """
    try:
        objs1 = get_final_state(plan1)
    except Exception as e:
        return False
    try:
        objs2 = get_final_state(plan2)
    except Exception as e:
        return False
    
    s1 = objs1.copy()
    s2 = objs2.copy()
    if any([o.sliced for o in objs1]):
        if not any([o.sliced for o in objs2]):
            return False
        for o1 in objs1:
            if 'knife' in o1.name.lower():
                s1.remove(o1)
            else:
                for o2 in objs2:
                    if o1 == o2:
                        s1.remove(o1)
    elif any([o.in_light for o in objs1]):
        if not any([o.in_light for o in objs2]):
            return False
        for o1 in objs1:
            if 'lamp' in o1.name.lower():
                s1.remove(o1)
                continue
            for o2 in objs2:
                if o1 == o2:
                    s1.remove(o1)
    else:
        for o1 in objs1:
            for o2 in objs2:
                if o1 == o2:
                    s1.remove(o1)
    return len(s1) == 0

def get_final_state(plan):
    in_hand = None
    seen_objs = []
    for i, action in enumerate(plan):
        if action[0] == 'PickupObject':
            assert in_hand == None
            temp = seen_objs
            for o in seen_objs:
                if o.name == action[1] and (o.recep == action[2] or (o.recep == None and action[2] == '')):
                    in_hand = o
                    temp.remove(o)
            seen_objs = temp
            if in_hand == None:
                in_hand = AlfredObject(action[1])
            in_hand.put(None)
        elif action[0] == 'SliceObject':
            assert 'knife' in in_hand.name.lower()
            obj = AlfredObject(action[1], recep=action[2])
            obj.slice()
            seen_objs.append(obj)
        elif action[0] == 'PutObject':
            assert action[1] == in_hand.name
            in_hand.put(action[2])
            seen_objs.append(in_hand)
            in_hand = None
        elif action[0] == 'ToggleObject':
            if 'lamp' in action[1].lower() and in_hand != None:
                raise Exception("Toggle object")
            in_hand.examine()
        elif action[0] == 'HeatObject':
            assert action[1] == in_hand.name or action[1] == ""
            in_hand.heat()
        elif action[0] == 'CoolObject':
            assert action[1] == in_hand.name or action[1] == ""
            in_hand.cool()
        elif action[0] == 'CleanObject':
            assert action[1] == in_hand.name or action[1] == ""
            in_hand.wash()
        else:
            raise Exception('Action not defined')

        # print("Action %s"%action)
        # for o in seen_objs:
        #     print(o)
        # print(" - in hand \n%s\n - "%in_hand)
        # print('-'*20)

    if in_hand != None:
        seen_objs.append(in_hand)

    # print("Final")
    # for o in seen_objs:
    #     print(o)
    # print(" - in hand \n%s\n - "%in_hand)
    # print('-'*20)

    return seen_objs

def is_plan_successful(traj_data, plan) -> bool:
    """
    Args:
        traj_data(dict)
        plan(list)
    Returns:
        success(bool)
    """
    seen_objs = get_final_state(plan)
    task_type = traj_data['task_type']
    pddl_param = traj_data['pddl_params']
    return _met_task_conditions(task_type, pddl_param, seen_objs)

def _met_task_conditions(task_type, params, objs):
    mrecep, target = [], []
    for o in objs:
        if o.name == params['mrecep_target']:
            mrecep.append(o)
        if o.name == params['object_target']:
            target.append(o)

    if params['object_sliced'] and not any([t.sliced for t in target]):
        return False
    if task_type == 'pick_and_place_simple'\
         and not any([t.recep == params['parent_target'] for t in target]):
        return False
    if task_type == 'pick_two_obj_and_place'\
         and not len([t.recep == params['parent_target'] for t in target]) > 1:
        return False
    if task_type == 'look_at_obj_in_light'\
        and not any([t.in_light for t in target]):
        return False
    if task_type == 'pick_clean_then_place_in_recep'\
        and not any([t.clean for t in target])\
             and not any([t.recep == params['parent_target'] for t in target]):
        return False
    if task_type == 'pick_heat_then_place_in_recep'\
        and not any([t.hot for t in target])\
             and not any([t.recep == params['parent_target'] for t in target]):
        return False
    if task_type == 'pick_cool_then_place_in_recep'\
        and not any([t.cold for t in target])\
             and not any([t.recep == params['parent_target'] for t in target]):
        return False
    if task_type == 'pick_and_place_with_movable_recep'\
        and not any([m.recep == params['parent_target'] for m in mrecep])\
             and not any([t.recep == params['mrecep_target'] for t in target]):
        return False
    return True

class AlfredObject:
    from data.alfred_data import constants
    receptacles = constants.RECEPTACLES
    val_recep_objects = constants.VAL_RECEPTACLE_OBJECTS
    val_action_objects = constants.VAL_ACTION_OBJECTS
    non_recep_objects = constants.NON_RECEPTACLES_SET

    def __init__(self, name, recep=None):
        self.sliced, self.clean, self.hot, self.cold, self.in_light = False, False, False, False, False
        assert name in AlfredObject.non_recep_objects
        self.name = name
        assert recep in AlfredObject.receptacles or recep == None or recep == ''
        if recep == '':
            self.recep = None
        else:
            self.recep = recep

    def __str__(self):
        desc = "%10s:\n"%(self.name)
        if self.sliced:
            desc += "\t%8s\n"%('sliced')
        if self.clean:
            desc += "\t%8s\n"%('clean')
        if self.hot:
            desc += "\t%8s\n"%('hot')
        if self.cold:
            desc += "\t%8s\n"%('cold')
        if self.in_light:
            desc += "\t%8s\n"%('in light')
        desc += "\t%8s"%('in %s'%self.recep)
        return desc

    def __eq__(self, other):
        if not isinstance(other, AlfredObject):
            return False
        return (self.name, self.recep, self.sliced, self.clean, self.hot, self.cold, self.in_light)\
             == (other.name, other.recep, other.sliced, other.clean, other.hot, other.cold, other.in_light) 

    def wash(self):
        assert self.name in AlfredObject.val_action_objects['Cleanable']
        self.clean = True

    def heat(self):
        assert self.name in AlfredObject.val_action_objects['Heatable']
        self.hot = True

    def cool(self):
        assert self.name in AlfredObject.val_action_objects['Coolable']
        self.cold = True

    def examine(self):
        assert self.name in AlfredObject.non_recep_objects
        self.in_light = True

    def put(self, recep):
        assert recep in AlfredObject.receptacles or recep == None
        self.recep = recep

    def slice(self):
        assert self.name in AlfredObject.val_action_objects['Sliceable']
        self.sliced = True

def tripletToString(plan:list) -> list:
    result = []
    for action in plan:
        result.append(get_action_description(action))
        if '\n' in result[-1]:
            print(plan)
    return result

def find_with_processed_goal(processed_goal, match):
    result = []
    for goal in match:
        if processed_goal == preprocess_goal(goal):
            result.append(match[goal])
    return result