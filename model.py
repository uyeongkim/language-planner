import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util as st_utils
from utils import plan_module, plan_helper
import pprint

class Vanilla:
    def __init__(self, config):
        self.split = config['args']['split']
        self.k = config['args']['num_ex']
        self.appended = config['args']['appended']
        appended = 'appended' if config['args']['appended'] else 'noappended'
        self.action_path = config['file_path'][appended]['available_actions']
        self.plan_args = {
            'temp': config['plan_args']['temp'],
            'stop': ':',
            'max_tokens': 1500,
            'n': config['plan_args']['num'],
            'presence_penalty': config['plan_args']['presence_penalty'],
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._set_sim_lm(config['args']['sim_lm'])
        self.examples = json.load(open(config['file_path'][appended]['examples'], 'r'))
        self.corpus_embdding = self.sim_lm.encode(list(self.examples.keys()), batch_size=512, convert_to_tensor=True, device=self.device)
        print("Model initialized")
        
    def _set_sim_lm(self, sim_lm):
        self.sim_lm = SentenceTransformer(sim_lm).to(self.device)
        word_embedding_model = self.sim_lm._first_module()
        word_embedding_model.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        word_embedding_model.auto_model.resize_token_embeddings(len(word_embedding_model.tokenizer))
        
    def _get_examples(self, goal):
        goal = plan_helper.preprocess_goal(goal)
        goal_embedding = self.sim_lm.encode(goal, convert_to_tensor=True, device=self.device)
        cos_scores = st_utils.pytorch_cos_sim(goal_embedding, self.corpus_embdding)[0].detach().cpu().numpy()
        sim_idxs = np.argsort(cos_scores)[-self.k:]
        goals = np.array(list(self.examples.keys()))[sim_idxs]
        scores = cos_scores[sim_idxs]
        plans = [self.examples[goal] for goal in goals]
        return goals, scores, plans
    
    # Only gives one prompt
    def _get_prompts(self, goal, s_goals, s_plans):
        sentences = s_goals
        plan_list = s_plans
        prompt = 'Make a plan to complete a given task\n\n'
        for i, example_task in enumerate(sentences):
            plans = plan_list[i] # This is a list of plans
            for plan in plans:
                prompt += '%s :\n'%example_task
                prompt += plan_helper.toString(plan)
                prompt += '\n'
        prompt += '%s :\n'%goal
        return [prompt]
        
    def get_plan(self, goal):
        pp = pprint.PrettyPrinter(indent=4)
        sim_goals, cos_scores, sim_plans = self._get_examples(goal)
        prompts = self._get_prompts(goal, sim_goals, sim_plans)
        
        text_plans_shape = (len(prompts)*self.plan_args['n'])
        plans_shape = (len(prompts)*self.plan_args['n'], 30, 3)
        score_shape = (len(prompts)*self.plan_args['n'], 30)
        probs_shape = (len(prompts)*self.plan_args['n'], self.plan_args['max_tokens'])
        text_plans, plans, scores, log_probs = np.zeros(text_plans_shape, dtype=np.dtype('U100')), -np.ones(plans_shape, dtype=np.dtype('int32')), -np.ones(score_shape), -np.ones(probs_shape)
        
        for p_idx, prompt in enumerate(prompts):
            response  = plan_module.get_gpt_response(prompt, self.plan_args)
            for c_idx, choice in enumerate(response.choices):
                pl_idx = p_idx*self.plan_args['n']+c_idx
                text_plans[pl_idx] = choice.text
                print('Warning: ResponseToPlan are not implemented yet')
                plan, score = plan_helper.gptResponseToAlfredPlan(choice.text, gpu_num=c_idx%torch.cuda.device_count())
                plans[p_idx*self.plan_args['n']+c_idx, :plan.shape[0], :] = plan
                scores[p_idx*self.plan_args['n']+c_idx, :score.shape[0]] = score
                log_probs[p_idx*self.plan_args['n']+c_idx, :len(choice.logprobs['token_logprobs'])] = np.array(choice.logprobs['token_logprobs'], dtype=np.float32)        
        
        uniq_plans, indexs, votes = np.unique(plans, axis=0, return_counts=True, return_inverse=True)
        uniq_plans = plan_helper.decode_plans(uniq_plans)
        uniq_text_plans = np.array([text_plans[np.where(indexs==i)] for i in range(votes.shape[0])])
        uniq_scores = np.array([scores[np.where(indexs==i)] for i in range(votes.shape[0])])
        uniq_log_probs = np.array([log_probs[np.where(indexs==i)] for i in range(votes.shape[0])])

        sort_idx = np.argsort(votes)[::-1]
        uniq_plans = np.array(uniq_plans)[sort_idx]
        uniq_text_plans = uniq_text_plans[sort_idx]
        uniq_scores = uniq_scores[sort_idx]
        uniq_log_probs = uniq_log_probs[sort_idx]
        votes = votes[sort_idx]
        
        return uniq_plans, votes, uniq_log_probs, uniq_scores, uniq_text_plans