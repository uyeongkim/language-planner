import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json
import os
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter()

GPU = 0
if torch.cuda.is_available():
    torch.cuda.set_device(GPU)
OPENAI_KEY = None  # replace this with your OpenAI API key, if you choose to use OpenAI API

source = 'huggingface'  # select from ['openai', 'huggingface']
translation_lm_id = 'stsb-roberta-large'  # see comments above for all options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = 0.8  # early stopping threshold based on matching score and likelihood score
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples

sampling_params = \
        {
            "max_tokens": 10,
            "temperature": 0.1,
            "top_p": 0.9,
            "num_return_sequences": 10,
            "repetition_penalty": 1.2,
            'use_cache': True,
            'output_scores': True,
            'return_dict_in_generate': True,
            'do_sample': True,
        }

# initialize Translation LM
translation_lm = SentenceTransformer(translation_lm_id).to(device)

# create action embeddings using Translated LM
with open('data/available_actions.json', 'r') as f:
    action_list = json.load(f)
action_list_embedding = translation_lm.encode(action_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

# create example task embeddings using Translated LM
with open('data/available_examples.json', 'r') as f:
    available_examples = json.load(f)
example_task_list = [example.split('\n')[0] for example in available_examples]  # first line contains the task name
example_task_embedding = translation_lm.encode(example_task_list, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory

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

with open('data/task2param.json', 'r') as f:
    task2param = json.load(f)

acc = {}
for split in ['valid_seen', 'valid_unseen']:
    match_scores = []
    pddl_scores = {}
    data_path = '../alfred/data/json_2.1.0/{sp}'.format(sp=split)
    for ep in os.listdir(data_path):
        for trial in os.listdir(os.path.join(data_path, ep)):
            with open(os.path.join(data_path, ep, trial, 'traj_data.json'), 'r') as f:
                data = json.load(f)
            anns = data['turk_annotations']['anns']
            pddl = data['pddl_params']
            for ann in anns:
                task = ann['task_desc']
                example_idx, matching_score = find_most_similar(task, example_task_embedding)
                # print(available_examples[example_idx])
                example = available_examples[example_idx].split('\n')[0].split(':')[1].strip()
                # print(example)
                accuray = calculate_pddl_similarity(pddl, task2param[example])
                match_scores.append(matching_score)
                for p, s in accuray.items():
                    if p in pddl_scores:
                        pddl_scores[p].append(s)
                    else:
                        pddl_scores[p] = [s]
    for p, s in pddl_scores.items():
        pddl_scores[p] = np.mean(s).astype(float)
    acc[split] = {'matching_acc': np.mean(matching_score).astype(float), 'pddl_acc': pddl_scores}

with open('result.json', 'w') as f:
    json.dump(acc, f, indent=4)