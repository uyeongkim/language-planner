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

def update_triplet(_input):
    task2plan, goal, plans = _input
    available_action_dict = json.load(open('data/available_actions_in_word.json', 'r'))
    temp = {}
    
    if goal in task2plan:
        return None
    result = []
    for plan in plans:
        result.append([s for action in plan for s in available_action_dict if available_action_dict[s] == action])
    return result

def main():
    train_triplet = json.load(open('data/triplet/train_tripletPlan.json', 'r'))
    if os.path.exists('data/plan/task2plan-triplet.json'):
        task2plan = json.load(open('data/plan/task2plan-triplet.json', 'r'))
    else:
        task2plan = {}

    key_list = [k for k in train_triplet.keys() if k not in task2plan.keys()]
    manager = Manager()
    task2plan = manager.dict(task2plan)
    for i in range(10):
        print('Iter %d'%i)
        start = int(len(key_list)/10*i)
        end = int(len(key_list)/10*(i+1))
        goals = key_list[start:end]
        inputs = [(task2plan, goal, train_triplet[goal]) for goal in goals]
        for i, ret in enumerate(parmap.map(update_triplet, inputs, pm_pbar=True, pm_processes=multiprocessing.cpu_count())):
            if ret == None:
                continue
            task2plan[goals[i]] = ret
        task2plan = dict(task2plan)
        json.dump(task2plan, open('data/plan/task2plan-triplet.json', 'w'), indent=4)
        
if __name__ == '__main__':
    main()
    