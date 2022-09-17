import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
import os
import argparse

with open('data/task2param.json', 'r') as f:
    TASK2PARAM = json.load(f)
with open('data/available_examples.json', 'r') as f:
    EXAMPLES = json.load(f)

# helper function for finding similar sentence in a corpus given a query
def find_most_similar(query_str, corpus_embedding):
    query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score

def calculate_pddl_similarity(p1, p2):
    pddl_scores = {}
    all = True
    for p, t in p1.items():
        pddl_scores[p] = p1[p] == p2[p]
        if all and not pddl_scores[p]:
            all = False
    pddl_scores['all'] = all
    return pddl_scores

def get_example_taskList():
    example_task_list = [example.split('\n')[0] for example in EXAMPLES]  # first line contains the task name
    return example_task_list

def find_best_pddl(task, example_task_embedding):
    example_idx, matching_score = find_most_similar(task, example_task_embedding)
    example = available_examples[example_idx].split('\n')[0].split(':')[1].strip()
    return TASK2PARAM[example]

def save_pddl_match(example_task_embedding):
    pddl_match = {}
    for split in ['valid_seen', 'valid_unseen']:
        data_path = '../alfred/data/json_2.1.0/{sp}'.format(sp=split)
        for ep in os.listdir(data_path):
            for trial in os.listdir(os.path.join(data_path, ep)):
                with open(os.path.join(data_path, ep, trial, 'traj_data.json'), 'r') as f:
                    data = json.load(f)
                anns = data['turk_annotations']['anns']
                pddl_match.update({ann['task_desc']: find_best_pddl(ann['task_desc'], example_task_embedding) for ann in anns})
    with open('pddl_match.json', 'w') as f:
        json.dump(pddl_match, f, indent=4)

if __name__ == '__main__':
    GPU = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU)

    parser = argparse.ArgumentParser()
    parser.add_argument('--translation_lm', type=str, default='stsb-roberta-large')
    # parser.add_argument('--max_steps', type=int, default=20, description='maximum number of steps to be generated')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize Translation LM
    try:
        translation_lm = SentenceTransformer(args.translation_lm).to(device)
    except Exception as e:
        print('model not in huggingface')
        print(e)
        exit()

    # create example task embeddings using Translated LM
    example_task_list = get_example_taskList()
    example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

    # save matching pddl_params
    save_pddl_match(example_task_embedding)
    
