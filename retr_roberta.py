"""
Find k most similar sentences and k/4 not much similar sentences from train data
for each requested sentences
"""

import argparse
import os
import re
import time
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import util as st_utils
from sentence_transformers import SentenceTransformer
import torch
from utils import plan_helper

def get_goal(traj_data, args):
    if args.appended:
        anns = [ann['task_desc']+'[SEP]'+' '.join(ann['high_descs']) for ann in traj_data['turk_annotations']['anns']]
    else:
        anns = [ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]
    processed_anns = [plan_helper.preprocess_goal(ann) for ann in anns]
    return processed_anns

def get_corpus_embedding(example_goal_list, lm, device):
    word_embedding_model = lm._first_module()
    word_embedding_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
    return lm.encode(example_goal_list, batch_size=512, convert_to_tensor=True, device=device)

def find_similar(query_embedding, corpus_embedding, args):
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    sim_idxs = np.argsort(cos_scores)[-args.k:]
    # not similar sentences are selected from -3k ~ -k
    not_sim_idx = np.random.choice(np.argsort(cos_scores)[-3*args.k:-args.k], int(args.k/4), replace=False)
    return sim_idxs, not_sim_idx, cos_scores

def update_match(match, file_path):
    if not os.path.exists(file_path):
        saved_match = {}
    else:
        print('Saved result file found')
        saved_match = json.load(open(file_path, 'r'))
    saved_match.update(match)
    json.dump(saved_match, open(file_path, 'w'), indent=4)
    return saved_match

def main(args):
    # param if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = SentenceTransformer('stsb-roberta-large').to(device)

    plan_example_path = 'data/plan/train%s_avail1.json'%('_appended' if args.appended else '') # processed goals
    if args.appended != ('appended' in plan_example_path):
        raise Exception("Goal exmaple file should have appended goals")
    print('Train data retrieved from %s'%plan_example_path)

    plan_example = json.load(open(plan_example_path, 'r'))
    example_goals = list(plan_example.keys())
    corpus_embedding = get_corpus_embedding(example_goals, lm, device)
    example_goals = np.array(example_goals)

    ep_cnt = 0
    for split in ['valid_seen', 'valid_unseen']:
        sentence_save_path = os.path.join('result', 'alfred', 'roberta', \
            '%s-sentence@%d-%d%s.json'%(split, args.k, int(args.k/4), '_appended' if args.appended else ''))
        data_path = 'data/alfred_data/json_2.1.0/{sp}'.format(sp=split)
        sentence_match = update_match({}, sentence_save_path)
        for ep in tqdm(os.listdir(data_path), desc=split):
            for trial in os.listdir(os.path.join(data_path, ep)):
                with open(os.path.join(data_path, ep, trial, 'traj_data.json'), 'r') as f:
                    traj_data = json.load(f)

                for i, goal in enumerate(get_goal(traj_data, args)):
                    goal_embedding = lm.encode(goal, convert_to_tensor=True, device=device)
                    sim_idxs, diff_idxs, cos_scores = find_similar(goal_embedding, corpus_embedding, args)
                    sentence_match[traj_data['turk_annotations']['anns'][i]['task_desc']] = \
                        {
                            "similar": {"goal": example_goals[sim_idxs].tolist(), \
                                "score": cos_scores[sim_idxs].tolist(), "plan": [plan_example[g] for g in example_goals[sim_idxs]]},
                            "diff": {"goal": example_goals[diff_idxs].tolist(), \
                                "score": cos_scores[diff_idxs].tolist(), "plan": [plan_example[g] for g in example_goals[diff_idxs]]}
                        }
                ep_cnt += 1
            if ep_cnt % 20 == 0:
                update_match(sentence_match, sentence_save_path)
        update_match(sentence_match, sentence_save_path)
        print('sentence match for %s saved in [%s]'%(split, sentence_save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--appended", action='store_true')
    args = parser.parse_args()
    main(args)