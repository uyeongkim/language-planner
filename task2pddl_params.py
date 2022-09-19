"""
This program predicts PDDL parameters for a given task description.
This is done by finding the most similar task description in the ALFRED train dataset.
"""

import json
import os
import argparse
from selectors import EpollSelector
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
from transformers import GPT2Tokenizer, GPT2Model

with open('data/task2param.json', 'r', encoding='utf-8') as f:
    TASK2PARAM = json.load(f)
with open('data/available_examples.json', 'r', encoding='utf-8') as f:
    EXAMPLES = json.load(f)
EXAMPLE_TASK_LIST = [example.split('\n')[0].split(':')[1].strip() for example in EXAMPLES]
with open('data/supported_sentence_transformer.txt', 'r', encoding='utf-8') as f:
    _SUPPORTED = f.read().splitlines()
SUPPORTED_MODEL = [model.split('\n')[0] for model in _SUPPORTED]

def find_most_similar(query_str, corpus_embedding, translation_lm, supported = True):
    """helper function for finding similar sentence in a corpus given a query"""
    if not supported:
        query_embedding = translation_lm(query_str)
    else:
        query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    most_similar_idx, matching_score = np.argmax(cos_scores), np.max(cos_scores)
    return most_similar_idx, matching_score

def calculate_pddl_similarity(p1, p2):
    """Returns if two PDDL parameters concide"""
    pddl_scores = {}
    all = True
    for p in p1:
        pddl_scores[p] = p1[p] == p2[p]
        if all and not pddl_scores[p]:
            all = False
    pddl_scores['all'] = all
    return pddl_scores

def find_best_pddl(task, example_task_embedding, translation_lm):
    if len(task) == 0:
        raise ValueError('Task description is empty')
    example_idx, _ = find_most_similar(task, example_task_embedding, translation_lm)
    example = EXAMPLE_TASK_LIST[example_idx]
    return TASK2PARAM[example]

def save_pddl_match(example_task_embedding, translation_lm, filename='data/pddl_match.json'):
    pddl_match = {}
    for split in ['valid_seen', 'valid_unseen']:
        data_path = '../alfred/data/json_2.1.0/{sp}'.format(sp=split)
        for ep in os.listdir(data_path):
            for trial in os.listdir(os.path.join(data_path, ep)):
                with open(os.path.join(data_path, ep, trial, 'traj_data.json'), 'r') as f:
                    data = json.load(f)
                anns = [ann['task_desc'] for ann in data['turk_annotations']['anns']]
                _anns = []
                for ann in anns:
                    if '\n' in ann:
                        _anns.extend(a.strip().replace('\b', '') for a in ann.split('\n') if a != '')
                    else:
                        _anns.append(ann.strip().replace('\b', ''))
                anns = _anns
                pddl_match.update({ann: find_best_pddl(ann, example_task_embedding, translation_lm) for ann in anns})
    with open(filename, 'w') as f:
        json.dump(pddl_match, f, indent=4)

def gpt_encoder(sentence):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2').to(device)
    model  = GPT2Model.from_pretrained('gpt2').to(device)
    token = tokenizer(sentence, return_tensors='pt').to(device)
    return model(**token)[0].mean(dim=1)

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
        word_embedding_model = translation_lm._first_module()
        word_embedding_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        # create example task embeddings using Translated LM
        example_task_embedding = translation_lm.encode(EXAMPLE_TASK_LIST, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory
        if args.translation_lm not in SUPPORTED_MODEL:
            with open('data/supported_sentence_transformer.txt', 'a') as f:
                f.write(args.translation_lm + '\n')
            SUPPORTED_MODEL.append(args.translation_lm)
    except NameError:
        print('model:{model} not in huggingface'.format(model=args.translation_lm))
        if args.translation_lm == 'gpt2':
            outputs = [gpt_encoder(task) for task in EXAMPLE_TASK_LIST]
            example_task_embedding = torch.cat(outputs, dim=0)
            translation_lm = gpt_encoder
        else:
            raise ValueError('model{model} not in huggingface'.format(model=args.translation_lm))

    # save matching pddl_params
    save_pddl_match(example_task_embedding, filename='data/gpt-pddl_match.json', translation_lm=translation_lm)
    