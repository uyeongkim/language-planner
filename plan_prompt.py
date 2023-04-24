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
from multiprocessing import Pool
from utils import plan_helper, config
import parmap
import pprint
import datetime
from utils import plan_module
import pickle

# Yonsei api key
openai.api_key = config.OPENAI['api_key']
openai.organization = config.OPENAI['organization']

def write_prompt(goal, sentences, plan_list):
    prompt = 'Make a plan to complete a given task\n\n'
    for i, example_task in enumerate(sentences):
        plans = plan_list[i] # This is a list of plans
        for plan in plans:
            prompt += '%s :\n'%example_task
            prompt += plan_helper.toString(plan)
            prompt += '\n'
    prompt += '%s :\n'%goal
    return prompt

def _is_prompt_successful(input):
    goal, prompt = input
    prompt += '%s :\n'%goal
    gpt_args = ARGS
    gpt_args['n'] = 3
    response = plan_helper.get_gpt_response(prompt, gpt_args)
    gt_plans = plan_helper.find_with_processed_goal(goal, TRAIN_GT)
    temp = []
    for plans in gt_plans:
        temp.extend(plans)
    gt_plans = temp
    success = False
    plans = []
    text = []
    for result in response.choices:
        text.append(result.text.strip())
        plan, _ = plan_helper.gptResponseToAlfredPlan(result.text, get_available_action_dict())
        plans.append(plan)
        if any([plan_helper.compare_plan(p, plan) for p in gt_plans]):
            success = True
            break
    print('\n[Generated for \'%s\']\n%s'%(goal, '\n\n'.join(text)))
    pp = pprint.PrettyPrinter()
    pp.pprint(plans)
    if success:
        print('Success')
    else:
        print('Fail')
    return success, plan_helper.count_plans(plans)

def test_prompt(sentences, plan_list, cos_scores, sim_test_goals, diff_test_goals, args):
    prompt = '\n'.join((write_prompt(list(sim_test_goals)[0], sentences, plan_list).split('\n')[:-2]))
    print(prompt)
    print('='*60)
    sim_rets =  parmap.map(_is_prompt_successful, [(goal, prompt) for goal in sim_test_goals], pm_pbar=True, pm_processes=os.cpu_count())
    diff_rets = parmap.map(_is_prompt_successful, [(goal, prompt) for goal in diff_test_goals], pm_pbar=True, pm_processes=os.cpu_count())
    sim_suc ,sim_cnt ,sim_c = 0, 0, []
    for success, count in sim_rets:
        if success:
            sim_suc += 1
        sim_c.append(count)
        sim_cnt += 1
    print('\n\n')
    print("%-20s: %02d / %02d : %.2f"%("Similar task ACC", sim_suc, sim_cnt, sim_suc/sim_cnt*100))
    print("%-20s: %.2f +/- %.2f"%("response count", np.mean(np.array(sim_c)), np.std(np.array(sim_c))))

    diff_suc ,diff_cnt ,diff_c = 0, 0, []
    for success, count in diff_rets:
        if success:
            diff_suc += 1
        diff_c.append(count)
        diff_cnt += 1
    print("%-20s: %02d / %02d : %.2f"%("Differnet task ACC", diff_suc, diff_cnt, diff_suc/diff_cnt*100))
    print("%-20s: %.2f +/- %.2f"%("response count", np.mean(np.array(diff_c)), np.std(np.array(diff_c))))
    cos_scores = np.array(cos_scores)
    print("%-20s: %.2f +/- %.2f"%("example cos scores", np.mean(cos_scores), np.std(cos_scores)))
    print('\n\n')
    return sim_suc/sim_cnt*100, diff_suc/diff_cnt*100, prompt

def get_prompts(goal, sentence_dict, args):
    """
    Args:
        sentence_dict: 
            {"similar": {"goal": [], "scores": [], "plan": []}, 
             "diff": {"goal": [], "scores": [], "plan": []}}
    Return:
        chosen prompt(str)
    """
    if args.vanilla:
        return [write_prompt(goal, sentence_dict['similar']['goal'], sentence_dict['similar']['plan'])]

    def _get_similar_info(sim_dict, splits):
        sentences = np.array([])
        plan_list = np.array([])
        cos_scores = np.array([])
        sim_goals, sim_plans, sim_scores = np.array(sim_dict["goal"]), np.array(sim_dict["plan"]), np.array(sim_dict["score"])
        for i in range(3):
            sentences = np.concatenate((sentences, sim_goals[splits[i]]), axis=0)
            plan_list = np.concatenate((plan_list, sim_plans[splits[i]]), axis=0)
            cos_scores = np.concatenate((cos_scores, sim_scores[splits[i]]), axis=0)
        return sentences, plan_list, cos_scores

    sim_idxs = np.arange(len(sentence_dict['similar']['goal']))
    np.random.seed(0)
    np.random.shuffle(sim_idxs)
    splits = np.split(sim_idxs, 4)
    prompts = []
    for sim_test_idx in range(4):
        sentences, plan_list, cos_scores = _get_similar_info(sentence_dict["similar"], [splits[i] for i in range(4) if i != sim_test_idx])
        print('-'*60)
        print('Prompt %d\n'%(sim_test_idx+1))
        sim_acc, diff_acc, prompt = test_prompt(sentences, plan_list, cos_scores, \
            np.array(sentence_dict["similar"]["goal"])[splits[sim_test_idx]], np.array(sentence_dict['diff']['goal']), args)
        if not (sim_acc == 0 and diff_acc == 0):
            prompts.append(prompt)
    return prompts

def get_available_action_dict():
    actions = json.load(open('data/available_actions.json', 'r'))
    return actions

# def roberta_penalty(roberta, gpt_logp):
#     avg_log_roberta = np.mean(np.log(np.array(roberta)))
#     return  gpt_logp + ROBERTA_COEFF*avg_log_roberta

# def choose_(prompts, cos_scores):
#     penalty = np.array([])
#     for i, prompt in enumerate(prompts):
#         gpt_logp = 0
#         for j, t in enumerate(prompt.logprobs.tokens):
#             if prompt.logprobs.tokens[j-1] == '\n' and not prompt.logprobs.tokens[j].isdigit():
#                 gpt_logp = sum(prompt.logprobs['token_logprobs'][:j])
#         if gpt_logp == 0:
#             gpt_logp = sum(prompt.logprobs['token_logprobs'])
#         penalty = np.append(penalty, np.array([roberta_penalty(cos_scores[i], gpt_logp)]))
#     return np.argsort(penalty)

def load_match(file_path, args):
    with open(file_path, 'r') as f:
        match = json.load(f)
        temp = {}
    return match

def update_match(match, file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            match = pickle.load(f)
    else:
        match = {}
    return match

# TODO: in hand filtering and continuity - in_hand probability를 곱해주면?
# TODO: presence_penalty를 음수로 주면 예시에 있던 단어를 더 많이 쓰도록 하므로 예시 밖의 단어를 가져올 확률이 적어짐
#       -- frequency_penalty는 등장한 비율로 하는데, 이건 다양한 plan을 만들어야 하는 입장에서 좋은 것 같지는 않음

# TRIPLET GT
# SEEN_GT = json.load(open('data/triplet/val_seen_tripletPlan.json', 'r'))
# UNSEEN_GT = json.load(open('data/triplet/val_unseen_tripletPlan.json', 'r'))
TRAIN_GT = json.load(open('data/triplet/train.json', 'r'))

ROBERTA_COEFF = 1
ARGS = {
    "temp": 0.9,
    "stop": ':', 
    "max_tokens": 2000
    }

def main(args):
    # Load sentence_match
    # sentence: [sim: [goals, scores], diff: [goals, scores]]
    sentence_match_file= 'result/alfred/roberta/%s-sentence@%d-%d.json'%(args.split, args.k, int(args.k/4))
    print('Sentence match loaded from [%s]'%sentence_match_file)
    method_desc = 'vanilla' if args.vanilla else 'penalty'
    save_folder = 'result/alfred/prompt%s/%s'%(args.k, method_desc)
    if os.path.exists(save_folder):
        files = [f for f in os.listdir(save_folder) if args.split in f]
        fid = len(files)-1 if args.resume else len(files)
    else:
        fid = 0
    save_file = os.path.join(save_folder, f'{args.split}_{fid}.p')
    print('Result file will be saved in [%s]'%save_file)
    if not os.path.exists(os.path.split(save_file)[0]):
        os.makedirs(os.path.split(save_file)[0], exist_ok=True)
    sentence_match = load_match(sentence_match_file, args)
    plan_match = update_match({}, save_file)
    pp = pprint.PrettyPrinter()
    
    print(f'Generate {args.n} Completions')
    ARGS['n'] = args.n

    # Start generating plan match
    for goal, sentence_dict in tqdm(sentence_match.items()):
        if goal in plan_match:
            continue
        print('\nGoal: %s\n'%goal)
        # choose best prompt out of 20 sentences
        prompts = get_prompts(goal, sentence_dict, args)
        n_gpu = torch.cuda.device_count()
        n_process = min(args.n, 10)
        
        # Get gpt3 response
        for prompt in prompts:
            # 그냥 prompts 받도록 수정
            response = plan_module.get_gpt_response(prompt, ARGS)
            pp.pprint([c.text for c in response.choices])
            print()
            packed_plan_score = parmap.starmap(plan_helper.gptResponseToAlfredPlan, \
                list(zip([c.text for c in response.choices], [pid%n_gpu for pid in range(n_process)])), \
                    pm_processes=n_process)
            packed_plan_score = np.array(packed_plan_score, dtype=object)
            try:
                cos_scores = packed_plan_score[:, 1]
                maxlen = max(len(r) for r in cos_scores)
                cos_scores = np.vstack([np.pad(np.array(r, dtype=np.float32), (0, maxlen-len(r)), \
                    'constant', constant_values=-1) for r in cos_scores])
                plans = packed_plan_score[:, 0]
                
                plans = plan_helper.encode_plans(plans)
            except Exception as e:
                print('Packed plan and score')
                pp.pprint(packed_plan_score)
                raise e
        
        uniq_plans, indexs, votes = np.unique(plans, axis=0, return_counts=True, return_inverse=True)
        num_plans = uniq_plans.shape[0]
        
        # TODO: 내 sorting method 만들어야 함
        sort_idx = np.argsort(votes)[::-1]
        prompt_probs = [sum(response.choices[i].logprobs['token_logprobs']) for i in range(args.n)]
        
        prompt_probs = np.exp(np.array(prompt_probs))
        plan_probs = np.zeros(votes.shape, dtype=np.float32)
        plan_cos_scores = np.zeros((votes.shape[0], cos_scores.shape[1]), dtype=np.float32)
        
        for i in range(num_plans):
            selection = np.argwhere(indexs == i).flatten()
            plan_probs[i] = np.sum(prompt_probs[selection])
            plan_cos_scores[i] = np.sum(cos_scores[selection, :], axis=0)

        # Process result string to dict
        uniq_plans = plan_helper.decode_plans(uniq_plans)
        plan_match[goal] = {
            "plan": [uniq_plans[i] for i in sort_idx],
            "vote": [votes[i] for i in sort_idx],
            "prompt_logp": [np.log(np.array(plan_probs))[i] for i in sort_idx],
            "trans_cossim": [plan_cos_scores[i] for i in sort_idx]
        }
        pp.pprint((goal, plan_match[goal]))
  
        # Save pddl match
        if len(list(plan_match.keys())) % 5 == 1:
            pickle.dump(plan_match, open(save_file, 'wb'))
            
        plan_match = update_match(plan_match, save_file)
        pickle.dump(plan_match, open(save_file, 'wb'))
    
    print('Saved Result in %s'%save_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--split', required=True)
    parser.add_argument('--k', type=int, required=True)
    parser.add_argument('--n', type=int, required=True)
    parser.add_argument('--s_idx', type=int, default=0)
    parser.add_argument('--e_idx', type=int, default=-1)

    # module
    parser.add_argument('--vanilla', action='store_true')
    parser.add_argument('--penalty', action='store_true')

    # run
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    main(args)