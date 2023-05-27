import json
import pickle
import os
import pprint
import argparse
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from utils.plan_module import Plan

def find_executable_plan(plans:list):
    """
    choose plan with most vosts from executables
    Args:
        plans (list): triplet plans
    """
    def remove_padded_action(triplets:list) -> list:
        return list(filter(['', '', ''].__ne__, triplets))
    executable = []
    for p_idx, trip_plan in enumerate(plans):
        try:
            plan = Plan(triplets=remove_padded_action(trip_plan))
        except:
            continue
        if plan.is_executable():
            executable.append(plan)
    return executable

def main(args):
    pp = pprint.PrettyPrinter()
    plan_data = pickle.load(open(args.eval_file, 'rb'))
    split = 'valid_unseen' if 'valid_unseen' in args.eval_file else 'valid_seen'
    root = f'data/alfred_data/json_2.1.0/{split}'
    
    cnt = 0
    suc_cnt = 0
    err_cnt = 0
    ft_cnt = 0
    for cc, goal in enumerate(plan_data):
        # find executable plans
        executable_plans = find_executable_plan(plan_data[goal]['plan'])
        
        # get gt traj_data
        for ep in os.listdir(root):
            for trial in os.listdir(os.path.join(root, ep)):    
                traj_data = json.load(open(os.path.join(root, ep, trial, 'traj_data.json'), 'r'))
                anns = [ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]
                if goal not in anns:
                    continue
                task_type = traj_data['task_type']
                pddl_param = traj_data['pddl_params']
        
        # if no executables
        if len(executable_plans) == 0:
            cnt += 1
            err_cnt += 1
            if args.verbose:
                print(cc, ':', end=' ')
                print(goal)
                print('No executable plans')
                pp.pprint(plan_data[goal]['plan'])
                print('\n'+task_type)
                pp.pprint(pddl_param)
                print('\n'+'-'*30+'\n')
            continue
        
        if args.max:
            suc = any([plan.is_plan_fulfilled(task_type, pddl_param) for plan in executable_plans])
        else:
            # just the first one
            best_plan = executable_plans[0]
            best_plan.high_desc = goal
            suc = best_plan.is_plan_fulfilled(task_type, pddl_param)
        
        if not suc and args.verbose:
            print(cc, ':', end=' ')
            print(goal)
            if not args.max:
                pp.pprint(best_plan.high_actions)
            print('\n'+task_type)
            pp.pprint(pddl_param)
            print('Failure')
            print('\n'+'-'*30+'\n')
        else:
            suc_cnt += 1

        cnt += 1
        
    print(f"Success rate: {suc_cnt} / {cnt} : {(suc_cnt/cnt)*100:.2f}")
    print(f"Error rate : {err_cnt} / {cnt} : {(err_cnt/cnt)*100:.2f}")
    
def main_2(args):
    pp = pprint.PrettyPrinter()
    plan_data = pickle.load(open(args.eval_file, 'rb'))
    split = 'valid_unseen' if 'valid_unseen' in args.eval_file else 'valid_seen'
    root = f'data/alfred_data/json_2.1.0/{split}'
    result_file = f"result/{args.eval_file.split('/')[2]}/{args.eval_file.split('/')[3]}/{split}.p"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    print("Result file: ", result_file)
    
    result_dict = OrderedDict()
    cnt, suc_cnt, err_cnt = 0, 0, 0
    for ep in tqdm(os.listdir(root)):
        for trial in os.listdir(os.path.join(root, ep)):
            traj_data = json.load(open(os.path.join(root, ep, trial, 'traj_data.json'), 'r'))
            task_type = traj_data['task_type']
            pddl_param = traj_data['pddl_params']
            
            goals = [ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]
            for g_idx, goal in enumerate(goals):
                executable_plans = find_executable_plan(plan_data[goal]['plan'])
                # no executable plans
                if len(executable_plans) == 0:
                    plan = Plan(triplets=plan_data[goal]['plan'][0], ignore_exception=True)
                    
                    high_idxs, low_actions = plan.get_low_actions()
                    seen_objs = [o for o in list(np.unique(np.array(low_actions)[:, 1])) if o != '']
                    result_dict[goal] =  OrderedDict({
                        "root": os.path.join(data, ep, trial, 'traj_data.json'),
                        "instr_natural": traj_data['turk_annotations']['anns'][g_idx]['high_descs'],
                        "lan_triplet": plan.high_actions,
                        "triplet": plan.get_ex_high(),
                        "low_actions": list(np.array(low_actions)[:, 0]),
                        "low_classes": list(np.array(low_actions)[:, 1]),
                        "high_idxs": high_idxs,
                        "seen_objs": seen_objs
                    })
                    cnt += 1
                    err_cnt += 1
                    if args.verbose:
                        print(cnt+1, ':', end=' ')
                        print(goal)
                        print('No executable plans')
                        pp.pprint(plan_data[goal]['plan'])
                        print('\n'+task_type)
                        pp.pprint(pddl_param)
                        print('\n'+'-'*30+'\n')
                    continue
                
                # just pick the first one
                best_plan = executable_plans[0]
                best_plan.high_desc = goal
                suc = best_plan.is_plan_fulfilled(task_type, pddl_param)

                high_idxs, low_actions = best_plan.get_low_actions()
                seen_objs = []
                for obj in best_plan.get_final_state():
                    seen_objs.append(obj.name)
                    
                for obj in best_plan.get_final_state():
                    if obj.recep is not None and obj.recep not in seen_objs:
                        seen_objs.append(obj.recep)
                    if obj.in_light:
                        for o in list(np.array(low_actions)[:, 1]):
                            if 'lamp' in o.lower():
                                lamp = o
                                break
                        seen_objs.append(lamp)
                    if obj.sliced:
                        seen_objs.append(obj.name + 'Sliced')
                        
                result_dict[goal] =  OrderedDict({
                    "root": os.path.join(root, ep, trial, 'traj_data.json'),
                    "instr_natural": traj_data['turk_annotations']['anns'][g_idx]['high_descs'],
                    "lan_triplet": best_plan.high_actions,
                    "triplet": best_plan.get_ex_high(),
                    "low_actions": list(np.array(low_actions)[:, 0]),
                    "low_classes": list(np.array(low_actions)[:, 1]),
                    "high_idxs": high_idxs,
                    "seen_objs": list(seen_objs)
                })
                
                if not suc and args.verbose:
                    print(cnt+1, ':', end=' ')
                    print(goal)
                    if not args.max:
                        pp.pprint(best_plan.high_actions)
                    print('\n'+task_type)
                    pp.pprint(pddl_param)
                    print('Failure')
                    print('\n'+'-'*30+'\n')
                else:
                    suc_cnt += 1
                cnt += 1
                
    pickle.dump(result_dict, open(result_file, 'wb'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_file', type=str)
    parser.add_argument('--max', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    print(f"Eval file: {args.eval_file}")
    main_2(args)