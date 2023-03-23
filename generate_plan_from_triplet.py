"""
Generate Dictionary matching [Train goal]:[Sentence described plan]
"""

import json
from utils import plan_helper
import os
from tqdm import tqdm, trange
import multiprocessing
from multiprocessing import Manager
import time
import random
import parmap
import numpy as np
import argparse
import utils.plan_module as planner
import pprint

def get_train_triplet(args):
    alfred_data_path = 'data/alfred_data/json_2.1.0/train'
    train_triplet = {}
    for ep in tqdm(os.listdir(alfred_data_path), desc='Getting Train Triplet'):
        for trial in os.listdir(os.path.join(alfred_data_path, ep)):
            traj_data = json.load(open(os.path.join(alfred_data_path, ep, trial, 'traj_data.json'), 'r'))
            for r_idx, ann in enumerate(traj_data['turk_annotations']['anns']):
                try:
                    plan = planner.Plan.from_traj(traj_data, r_idx)
                except Exception as e:
                    print(os.path.join(ep, trial))
                    raise e
                if args.debug:
                    print(ep)
                    pp = pprint.PrettyPrinter()
                    pp.pprint(plan.high_actions)
                    print()
                if args.appended:
                    goal = f"{ann['task_desc']}[SEP]{' '.join(ann['high_descs'])}"
                else:
                    goal = ann['task_desc']
                if goal in train_triplet:
                    train_triplet[goal].append(plan.high_actions)
                else:
                    train_triplet[goal] = [plan.high_actions]
    return train_triplet

def update_triplet(_input):
    """upadte task2plan with one goal and plans in string sentence form"""
    task2plan, goal, plans = _input
    available_action_dict = json.load(open('data/available_actions_in_word2.json', 'r'))
    temp = {}
    
    if goal in task2plan:
        return None
    result = []
    for plan in plans:
        result.append([s for action in plan for s in available_action_dict if available_action_dict[s] == action])
    return result

def main(args):
    # paths
    train_path = 'data/triplet/train_appended.json'
    result_path = 'data/plan/train_appended_avail2.json'

    if not os.path.exists(train_path):
        goal2triplet = get_train_triplet(args)
        json.dump(goal2triplet, open(train_path, 'w'), indent=4)
    else:
        goal2triplet = json.load(open(train_path, 'r'))
    goal2sPlan = json.load(open(result_path, 'r')) if os.path.exists(result_path) else {} # resume

    key_list = [k for k in goal2triplet.keys() if k not in goal2sPlan.keys()]
    manager = Manager()
    goal2sPlan = manager.dict(goal2sPlan)
    for i in range(10):
        print('Iter %d'%i)
        start = int(len(key_list)/10*i)
        end = int(len(key_list)/10*(i+1))
        goals = key_list[start:end]
        inputs = [(goal2sPlan, goal, goal2triplet[goal]) for goal in goals]
        for i, ret in enumerate(parmap.map(update_triplet, inputs, pm_pbar=True, pm_processes=multiprocessing.cpu_count())):
            if ret == None:
                continue
            goal2sPlan[goals[i]] = ret
        goal2sPlan = dict(goal2sPlan)
        json.dump(goal2sPlan, open(result_path, 'w'), indent=4)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--appended', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    main(args)
    