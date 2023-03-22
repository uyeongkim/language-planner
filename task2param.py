"""
This program predicts PDDL parameters for a given task description.
This is done by finding the most similar task description in the ALFRED train dataset.
"""

import json
import os
import argparse
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
from transformers import GPT2Tokenizer, GPT2Model
import re
from tqdm import tqdm
import random
import pickle
from multiprocessing import Manager
from multiprocessing import Pool
import multiprocessing
import time

def preprocess(s):
    # remove escape sequence
    s = s.strip()
    if "\n" in s:
        s = s.split("\n")[0]
    elif "\b" in s:
        s = s.strip("\b")
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
            left = ''
    if len(ls) == 0:
        ls = left.split()
    # lower
    return ' '.join(ls).lower().replace('.', '')

###### task2param data #############################
# with open('data/fullGeneratedTask2Param.json', 'r', encoding='utf-8') as f:
#     TASK2PARAM = json.load(f)
with open('data/newTask2Param.json', 'r', encoding='utf-8') as f:
    TASK2PARAM = json.load(f)
# with open('data/generatedTask2Param.json', 'r') as f:
#     _task2param = json.load(f)
#     TASK2PARAM.update(_task2param)
temp = {}
for t in TASK2PARAM:
    temp[preprocess(t)] = TASK2PARAM[t]
TASK2PARAM = temp
####################################################

EXAMPLE_TASK_LIST = list(TASK2PARAM.keys())

def find_most_similar(query_str, corpus_embedding, translation_lm, k, supported = True):
    """helper function for finding similar sentence in a corpus given a query"""
    if not supported:
        query_embedding = translation_lm(query_str)
    else:
        query_embedding = translation_lm.encode(query_str, convert_to_tensor=True, device=device)
    # calculate cosine similarity against each candidate sentence in the corpus
    cos_scores = st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()
    # retrieve high-ranked index and similarity score
    idx = np.argsort(cos_scores)[-k:]
    most_similar_idx, matching_score = idx, cos_scores[idx]
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

def find_best_pddl(task, example_task_embedding, translation_lm, k):
    if len(task) == 0:
        raise ValueError('Task description is empty')
    example_idx, _ = find_most_similar(task, example_task_embedding, translation_lm, k)
    example = [EXAMPLE_TASK_LIST[i] for i in example_idx]
    return [TASK2PARAM[e] for e in example], dict(zip(['found', 'original'], [example, task]))

def save_pddl_match(example_task_embedding, args, translation_lm=None, filename='data/pddl_match.json'):
    pddl_match = {}
    sentence_match = dict(zip(['found', 'original'], [[], []]))
    for split in ['valid_seen', 'valid_unseen']:
        data_path = '../alfred/data/json_2.1.0/{sp}'.format(sp=split)
        for ep in tqdm(os.listdir(data_path)):
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
                anns = [preprocess(ann) for ann in _anns]
                for i, ann in enumerate(anns):
                    pddls, new_sentence_match = find_best_pddl(ann, example_task_embedding, translation_lm, args.k)
                    pddl_match[_anns[i]] = list(pddls)
                    for k in new_sentence_match:
                        sentence_match[k].append(new_sentence_match[k])
    with open(filename, 'w') as f:
        json.dump(pddl_match, f, indent=4)
    return sentence_match

def gpt_encoder(sentence):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2').to(device)
    model  = GPT2Model.from_pretrained('gpt2').to(device)
    token = tokenizer(sentence, return_tensors='pt').to(device)
    return model(**token)[0].mean(dim=1)

def cosine_similarity(query_embedding, corpus_embedding):
    return st_utils.pytorch_cos_sim(query_embedding, corpus_embedding)[0].detach().cpu().numpy()

def get_embedding(str, engine, embeddings=None, embedding_cache=None): 
    if embeddings is None:
        # query embedding
        if embedding_cache is not None and str in embedding_cache:
            if type(embedding_cache[str][-1]) == bool:
                embedding_cache[str] = embedding_cache[str][0]
            return embedding_cache[str] if type(embedding_cache[str]) == np.array else np.array(embedding_cache[str]).reshape((1, -1))
        time.sleep(1/20)
        return np.array(openai.Embedding.create(input=[str], engine=engine)["data"][0]['embedding']).reshape((1, -1))
    else:
        # corpus embedding
        if embedding_cache is not None and str in embedding_cache:
            if type(embedding_cache[str][-1]) == bool:
                embedding_cache[str] = embedding_cache[str][0]
            return embedding_cache[str] if type(embedding_cache[str]) == np.array else np.array(embedding_cache[str]).reshape((1, -1))
        time.sleep(1/20)
        embeddings[str] = np.array(openai.Embedding.create(input=[str], engine=engine)["data"][0]['embedding']).reshape((1, -1))

def get_task2param(corpus, debug):
    if 'template' in corpus:
        task2param = json.load(open('data/fullGeneratedTask2Param.json', 'r', encoding='utf-8'))
    elif 'alfred' in corpus:
        task2param = json.load(open('data/task2param.json', 'r', encoding='utf-8'))
        if 'no gen' not in corpus:
            augmentation = json.load(open('data/generatedTask2Param.json', 'r'))
            task2param.update(augmentation)
    else:
        raise Exception('corpus not specified')
    temp = {}
    for t in task2param:
        if preprocess(t) == '':
            raise Exception('goal became empty after preprocess\n%s'%t)
        temp[preprocess(t)] = task2param[t]
    if debug:
        keys = random.sample(list(temp.keys()), 5)
        return {k:temp[k] for k in keys}
    return temp

# def find_pddl_match(query_str, corpus_embeddings, embedding_cache, args):
#     task2param = get_task2param(args.corpus, args.debug)
#     example_task_list = list(task2param.keys())
#     query_embedding, is_new = get_embedding(query_str, args.translation_lm, embedding_cache=embedding_cache)
#     cos_score = cosine_similarity(query_embedding, corpus_embeddings)
#     indices = np.argsort(cos_score)[-args.k:]
#     if args.verbose:
#         print('The most similar sentence with "{}" is "{}"'.format(query_str, example_task_list[indices[-1]]))
#     if not is_new:
#         query_embedding = None
#     return [task2param[example_task_list[i]] for i in indices], query_embedding

def preprocess_annotation(str):
    if '\n' in str:
        for a in str.split('\n'):
            if a.strip().replace('\b', '') != '':
                return preprocess(a.strip().replace('\b', ''))
        raise Exception("Given annotation cannot be processed")
    else:
        return preprocess(str.strip().replace('\b', ''))

# def _save_pddl_match(corpus_embeddings, traj_data_path, pddl_match, embedding_cache, args):
#     with open(traj_data_path, 'r') as f:
#         data = json.load(f)
#     anns = [ann['task_desc'] for ann in data['turk_annotations']['anns'] if ann['task_desc'] not in pddl_match]
#     new_embed = dict()
#     for ann in anns:
#         processed_ann = preprocess_annotation(ann)
#         pddl_list, query_embedding = find_pddl_match(processed_ann, corpus_embeddings, embedding_cache, args)
#         pddl_match[ann] = pddl_list
#         if query_embedding is not None:
#             new_embed[processed_ann] = query_embedding
#     return pddl_match, new_embed

def get_pddl_list(query_embedding, corpus_embeddings, args):
    task2param = get_task2param(args.corpus, args.debug)
    example_task_list = list(task2param.keys())
    print("### in get pddl list")
    print("query_embedding shape", query_embedding.shape)
    print("corpus_embedding shape", corpus_embeddings.shape)
    cos_score = cosine_similarity(query_embedding, corpus_embeddings)
    indices = np.argsort(cos_score)[-args.k:]
    return [task2param[example_task_list[i]] for i in indices]

def get_corpus_embedding(corpus, args):
    task2param = get_task2param(corpus, args.debug)
    example_task_list = list(task2param.keys())
    if os.path.exists('data/{}_embedding.npy'.format(corpus)):
        return np.load('data/{}_embedding.npy'.format(corpus))
    try:
        embedding_cache = pickle.load(open('data/embedding.pkl', 'rb'))
    except:
        embedding_cache = {}

    print("Getting corpus embedding: MP start")
    pool = Pool(15)
    m = Manager()
    newly_found_embedding = m.dict()
    pool.starmap(get_embedding, [(e, args.translation_lm, newly_found_embedding, embedding_cache) for e in example_task_list])
    pool.close()
    pool.join()

    result = np.array([])
    for e in example_task_list:
        if e in newly_found_embedding:
            if type(newly_found_embedding[e]) != np.array:
                newly_found_embedding[e] = np.array(newly_found_embedding[e]).reshape((1, -1))
            result = np.concatenate((result, newly_found_embedding[e])) if result.size != 0 else newly_found_embedding[e]
        else:
            if type(embedding_cache[e]) != np.array:
                embedding_cache[e] = np.array(embedding_cache[e]).reshape((1, -1))
            if type(embedding_cache[e][0, 1]) == bool:
                embedding_cache[e] = embedding_cache[e][0, 0]
            result = np.concatenate((result, embedding_cache[e])) if result.size != 0 else embedding_cache[e]
    if True:
    # if len(dict(newly_found_embedding).keys()) != 0:
        embedding_cache.update(newly_found_embedding)
        pickle.dump(embedding_cache, open('data/embedding.pkl', 'wb'))
    
    return result

def print_spent_time(work, time):
    hr = int(time/(60*60))
    m = int((time-(60*60)*hr)/(60))
    s = time%60
    print("Time spent {}: {:3d}hrs {:3d}mins {:3.1f} secs".format(work, hr, m, s))

def save_pddl_match_gpt3(args):
    start = time.time()
    corpus_embedding = get_corpus_embedding(args.corpus, args)
    if args.verbose:
        print('Corpus embedding generated')
    mid = time.time()
    print_spent_time("getting corpus embedding", mid-start)

    # matching
    for split in ['valid_seen', 'valid_unseen']:
        data_path = '../alfred/data/json_2.1.0/{sp}'.format(sp=split)
        if args.debug:
            eps = tqdm(os.listdir(data_path)[:3])
        else:
            eps = tqdm(os.listdir(data_path))
        eps.set_postfix_str("episode in %s"%split)
        try:
            pddl_match = json.load(open(args.save_file, 'r'))
        except FileNotFoundError:
            pddl_match = {}
        for ep in eps:
            try:
                embedding_cache = pickle.load(open('data/embedding.pkl', 'rb'))
            except:
                embedding_cache = {}
            for trial in os.listdir(os.path.join(data_path, ep)):
                with open(os.path.join(data_path, ep, trial, 'traj_data.json'), 'r') as f:
                    traj_data = json.load(f)
                anns = [ann['task_desc'] for ann in traj_data['turk_annotations']['anns'] if ann['task_desc'] not in pddl_match]
                new_ann = [ann for ann in anns if preprocess_annotation(ann) not in embedding_cache]
                p = Pool(8)
                ret = p.starmap(get_embedding,[(a, args.translation_lm) for a in new_ann])
                p.close()
                p.join()
                if len(ret) != 0:
                    embedding_cache.update(dict(zip([preprocess_annotation(a) for a in new_ann], ret)))
                    pickle.dump(embedding_cache, open('data/embedding.pkl', 'wb'))
                embeddings = np.array([])
                for ann in anns:
                    pann = preprocess_annotation(ann)
                    if pann in embedding_cache:
                        if type(embedding_cache[pann][-1]) == bool:
                            embedding_cache[pann] = embedding_cache[pann][0]
                        embedding = np.array(embedding_cache[pann]).reshape((1, -1))
                        embeddings = np.concatenate((embeddings, embedding)) if embeddings.size != 0 else embedding
                    else:
                        embeddings = np.concatenate((embeddings, ret.pop(0))) if embeddings.size != 0 else ret.pop(0)
                p = Pool(8)
                ret = p.starmap(get_pddl_list, [(e, corpus_embedding, args) for e in embeddings])
                p.close()
                p.join()
                pddl_match.update(dict(zip(anns, ret)))
        json.dump(pddl_match, open(args.save_file, 'w'), indent=4)
    end = time.time()
    print_spent_time("matching validation annotations", end-mid)
    print_spent_time("in total", end-start)
    os.makedirs(os.path.split(args.save_file)[0], exist_ok=True)
    json.dump(pddl_match, open(args.save_file, 'w'), indent=4)
    if args.verbose:
        print('saved file to {}'.format(args.save_file))

if __name__ == '__main__':
    GPU = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(GPU)

    parser = argparse.ArgumentParser()
    parser.add_argument('--translation_lm', type=str, default='stsb-roberta-large')
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--save_file', type=str) # only for gpt3
    parser.add_argument('--corpus', type=str, choices=['alfred', 'template', 'alfred no gen'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.translation_lm in ['text-similarity-baddage-001', 'text-similarity-davinci-001']:
        import openai
        openai.api_key = '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'
        openai.organization = 'org-azdthpxrguDHQc2ujvxf4hTZ'
        save_pddl_match_gpt3(args)
    else:
        # initialize Translation LM
        try:
            translation_lm = SentenceTransformer(args.translation_lm).to(device)
            word_embedding_model = translation_lm._first_module()
            word_embedding_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
            # create example task embeddings using Translated LM
            example_task_embedding = translation_lm.encode(EXAMPLE_TASK_LIST, batch_size=512, convert_to_tensor=True, device=device)  # lower batch_size if limited by GPU memory
        except NameError:
            raise ValueError('model{model} not in huggingface'.format(model=args.translation_lm))

        # save matching pddl_params
        sentence_match = save_pddl_match(example_task_embedding, args, filename=args.save_file, translation_lm=translation_lm)
        with open(args.save_file.replace('pddl', 'sentence'), 'w') as f:
            json.dump(sentence_match, f, indent=4)
    