from utils.plan_module import Plan
import json
import pickle
import os
import pprint
import argparse

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
        
        # just the first one
        best_plan = executable_plans[0]
        best_plan.high_desc = goal
        
        suc = best_plan.is_plan_fulfilled(task_type, pddl_param)
                
        if not suc and args.verbose:
            print(cc, ':', end=' ')
            print(goal)
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
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_file', type=str)
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    print(f"Eval file: {args.eval_file}")
    main(args)