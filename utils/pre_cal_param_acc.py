import os
import json
import argparse
import numpy as np
import re
import string

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
    # lower
    return ' '.join(ls).lower().replace('.', '')

def film_preprocess(s):
    return ''.join([c for c in s.lower() if c not in set(string.punctuation)])

def process_task(raw_tasks):
    tasks = []
    for t in raw_tasks:
        if '\n' in t:
            tasks.extend([_t.replace('\b', '').strip() for _t in t.split('\n') if _t != ''])
        else:
            tasks.append(t.replace('\b', '').strip())
    return tasks

def print_goal(task):
    log = '='*80+"\n"
    log += task+"\n"+"="*80+"\n"
    return log

def print_subgoal(subgoal):
    log = 'Subgoal:\n'
    for j, s in enumerate(subgoal):
        log += '%2d: %s\n'%(j, s)
    log += "="*80+"\n"
    return log

def print_sentence_match(found, real_match):
    if len(found) > 5:
        log = 'Found sentence: {}\n'.format('\n'.join(found[:5]))
        log += '...\n'
    else:
        log = 'Found sentence: {}\n'.format('\n'.join(found[:-1]))
        
    if len(real_match) > 0:
        log += '\nReal matching goals in task2param was...\n' 
    for i, r in enumerate(real_match):
        if i > 4:
            break
        log += '%2d: %s\n'%(i, r)
    log += "="*80+"\n"
    return log

def print_table(gtPddl, pred):
    log = "{:20} {:^20} | {:^20}\n".format(' ', 'gt', 'pred')
    task_type_repr = {
        'look_at_obj_in_light': 'look',
        'pick_heat_then_place_in_recep': 'heat',
        'pick_cool_then_place_in_recep': 'cool',
        'pick_and_place_with_movable_recep': 'mrecep',
        'pick_two_obj_and_place': 'two',
        'pick_clean_then_place_in_recep': 'clean',
        'pick_and_place_simple': 'simple'
    }
    for k in gtPddl:
        if k not in pred:
            log += "{:<20}: {:<20}|{:<20}{}\n".format(k, gtPddl[k], "NO PRED", '*')
            continue
        if 'sliced' in k:
            if type(pred[k]) != type(True):
                log += "{:<20}: {:<20}|{:<20}{}\n".format(k, 'True' if gtPddl[k] else 'False', pred[k], '*' if gtPddl[k] != pred[k] else '')
            else:
                log += "{:<20}: {:<20}|{:<20}{}\n".format(k, 'True' if gtPddl[k] else 'False', 'True' if pred[k] else 'False', '*' if gtPddl[k] != pred[k] else '')
        elif 'task_type' in k:
            if pred[k] not in task_type_repr:
               log += "{:<20}: {:<20}|{:<20}{}\n".format(k, gtPddl[k], pred[k], '*' if gtPddl[k] != pred[k] else '')
            else:
                log += "{:<20}: {:<20}|{:<20}{}\n".format(k, task_type_repr[gtPddl[k]], task_type_repr[pred[k]], '*' if gtPddl[k] != pred[k] else '')
        else:    
            log += "{:<20}: {:<20}|{:<20}{}\n".format(k, gtPddl[k], pred[k], '*' if gtPddl[k] != pred[k] else '')
    log += "\n"
    return log

def ans2x(ans):
    return ans.lower().replace('.', '').replace(',', '').replace('\'', '').replace('-', '').replace('/', '')

def print_acc(all_correct, type_correct, m_correct, o_correct, s_correct, p_correct, t_correct, total):
    log = "="*80+"\n"
    log += "Total annotations: {:<4}, All correct annotation: {:<4}\n".format(total, all_correct)
    log += "All    correct SR: {:.2f}\n".format(all_correct/total*100)
    log += "type   correct SR: {:.2f}\n".format(type_correct/total*100)
    log += "mrecep correct SR: {:.2f}\n".format(m_correct/total*100)
    log += "object correct SR: {:.2f}\n".format(o_correct/total*100)
    log += "sliced correct SR: {:.2f}\n".format(s_correct/total*100)
    log += "parent correct SR: {:.2f}\n".format(p_correct/total*100)
    log += "toggle correct SR: {:.2f}\n".format(t_correct/total*100)
    return log

def main(args):
    code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(code_path, 'data', 'alfred_data', 'json_2.1.0')
    with open(args.pddl_match_file, 'r') as f:
        prediction_pddl_match = json.load(f)
    if args.print_sentence_match:
        if os.path.exists(args.sentence_match_file):
            with open(args.sentence_match_file, 'r') as f:
                # {original: [s], found: [[]]}
                sentence_match = json.load(f)
        else:
            with open(args.pddl_match_file.replace('pddl', 'sentence'), 'r') as f:
                # {original: [s], found: [[]]}
                sentence_match = json.load(f)
        with open(args.t2pData, 'r') as f:
            task2param = json.load(f)

    if args.split == 'all':
        splits = ['valid_seen', 'valid_unseen']
    elif args.split == 'seen':
        splits = ['valid_seen']
    else:
        splits = ['valid_unseen']

    all_correct = 0
    type_correct = 0
    m_correct = 0
    o_correct = 0
    s_correct = 0
    p_correct = 0
    t_correct = 0
    cnt = 0
    log = ""
    
    for split in splits:
        for ep in os.listdir(os.path.join(data_path, split)):
            if os.path.isfile(os.path.join(data_path, split, ep)):
                continue
            for trial in os.listdir(os.path.join(data_path, split, ep)):
                with open(os.path.join(data_path, split, ep, trial, 'traj_data.json')) as f:
                    traj_data = json.load(f)
                gtPddl = traj_data['pddl_params']
                gtPddl['task_type'] = traj_data['task_type']
                if args.film_preprocess:
                    tasks = [film_preprocess(ann['task_desc']) for ann in traj_data['turk_annotations']['anns']]
                else:
                    # tasks = process_task([ann['task_desc'] for ann in traj_data['turk_annotations']['anns']])
                    tasks = [ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]
                subgoals = [ann['high_descs'] for ann in traj_data['turk_annotations']['anns']]
                # predicted_pddl = [[첫번째 task에 대한 pddl들] ... ]
                predicted_pddl = []
                for t in tasks:
                    try:
                        predicted_pddl.append(prediction_pddl_match[t])
                    except KeyError:
                        for k in prediction_pddl_match:
                            if t in k:
                                raw_goal = k
                        predicted_pddl.append(prediction_pddl_match[raw_goal])
                for i, _p in enumerate(predicted_pddl):
                    # _p는 첫번째 task instruciton에 대한 prediction들
                    if type(_p) == dict:
                        if gtPddl != _p:
                            log += print_goal(tasks[i])
                            if args.print_subgoal:
                                log += print_subgoal(subgoals[i])
                            if args.print_sentence_match:
                                found = sentence_match['found'][sentence_match['original'].index(preprocess(tasks[i]))]
                                real_match = [t for t in task2param if task2param[t] == gtPddl]
                                log += print_sentence_match(found, real_match)
                                # find if right pddl config already exists but it didn't fount one
                            if args.print_table:
                                log += print_table(gtPddl, _p)
                    else:
                        for p in _p:
                            if gtPddl != p:
                                log += print_goal(tasks[i])
                                if args.print_subgoal:
                                    log += print_subgoal(subgoals)
                                if args.print_sentence_match:
                                    found = sentence_match['found'][sentence_match['original'].index(preprocess(tasks[i]))]
                                    real_match = [t for t in task2param if task2param[t] == gtPddl]
                                    log += print_sentence_match(found, real_match)
                                if args.print_table:
                                    log += print_table(gtPddl, p)
                
                for pddls in predicted_pddl:
                    idx = np.where(np.array(pddls) == gtPddl, True, False)
                    cnt += len(pddls) if type(pddls) == list else 1
                    if type(pddls) == dict and pddls != gtPddl:
                        # @1 prediction and wasn't correct
                        type_correct = type_correct+1 if 'task_type' in pddls and gtPddl['task_type'] == pddls['task_type'] else type_correct 
                        m_correct  = m_correct+1 if 'mrecep_target' in pddls and gtPddl['mrecep_target'] == pddls['mrecep_target'] else m_correct
                        o_correct  = o_correct+1 if 'object_target' in pddls and gtPddl['object_target'] == pddls['object_target'] else o_correct
                        s_correct  = s_correct+1 if 'object_sliced' in pddls and gtPddl['object_sliced'] == pddls['object_sliced'] else s_correct
                        p_correct  = p_correct+1 if 'parent_target' in pddls and gtPddl['parent_target'] == pddls['parent_target'] else p_correct
                        if 'toggle_target' in pddls:
                            t_correct  = t_correct+1 if gtPddl['toggle_target'] == pddls['toggle_target'] else t_correct
                        else:
                            pred_toggle = 'floorLamp' if pddls['task_type'] == 'look_at_obj_in_light' else ''
                            t_correct  = t_correct+1 if gtPddl['toggle_target'] == pred_toggle else t_correct
                    
                    elif type(pddls) == list and len(pddls) == 1 and not idx[0]:
                        # @1 prediction and wasn't correct
                        type_correct = type_correct+1 if 'task_type' in pddls[0] and gtPddl['task_type'] == pddls[0]['task_type'] else type_correct 
                        m_correct  = m_correct+1 if 'mrecep_target' in pddls[0] and gtPddl['mrecep_target'] == pddls[0]['mrecep_target'] else m_correct
                        o_correct  = o_correct+1 if 'object_target' in pddls[0] and gtPddl['object_target'] == pddls[0]['object_target'] else o_correct
                        s_correct  = s_correct+1 if 'object_sliced' in pddls[0] and gtPddl['object_sliced'] == pddls[0]['object_sliced'] else s_correct
                        p_correct  = p_correct+1 if 'parent_target' in pddls[0] and gtPddl['parent_target'] == pddls[0]['parent_target'] else p_correct
                        t_correct  = t_correct+1 if 'toggle_target' in pddls[0] and gtPddl['toggle_target'] == pddls[0]['toggle_target'] else t_correct

                    elif idx.ndim != 0 and sum(idx) == 0:
                        # @K prediction and no prediction was correct
                        c = []
                        for p in pddls:
                            cnt = 0
                            for k in gtPddl:
                                if gtPddl[k] == p[k]:
                                    cnt += 1
                        c.append(cnt)
                        idx = np.argmax(c)
                        m_correct  = m_correct+1 if gtPddl['mrecep_target'] == pddls[idx]['mrecep_target'] else m_correct
                        type_correct  = type_correct+1 if gtPddl['task_type'] == pddls[idx]['task_type'] else type_correct
                        o_correct  = o_correct+1 if gtPddl['object_target'] == pddls[idx]['object_target'] else o_correct
                        s_correct  = s_correct+1 if gtPddl['object_sliced'] == pddls[idx]['object_sliced'] else s_correct
                        p_correct  = p_correct+1 if gtPddl['parent_target'] == pddls[idx]['parent_target'] else p_correct
                        t_correct  = t_correct+1 if gtPddl['toggle_target'] == pddls[idx]['toggle_target'] else t_correct
                        
                    else:
                        all_correct += 1
                        type_correct += 1
                        m_correct += 1
                        o_correct += 1
                        s_correct += 1
                        p_correct += 1
                        t_correct += 1
                log += "\n"

    print('All correct: %d / %d : %.2f'%(all_correct, cnt, (all_correct/cnt*100)))
    print('type correct: %d / %d : %.2f'%(type_correct, cnt, (type_correct/cnt*100)))
    print('mrecep correct: %d / %d : %.2f'%(m_correct, cnt, (m_correct/cnt*100)))
    print('obejct correct: %d / %d : %.2f'%(o_correct, cnt, (o_correct/cnt*100)))
    print('sliced correct: %d / %d : %.2f'%(s_correct, cnt, (s_correct/cnt*100)))
    print('parent correct: %d / %d : %.2f'%(p_correct, cnt, (p_correct/cnt*100)))
    print('toggle correct: %d / %d : %.2f'%(t_correct, cnt, (t_correct/cnt*100)))
                        
    if args.print_acc:
        if args.split == 'all':
            total = 820+820
        elif args.split == 'seen':
            total = 820
        else:
            total = 821

        log += print_acc(all_correct, type_correct, m_correct, o_correct, s_correct, p_correct, t_correct, total)
    if not os.path.exists(os.path.split(args.outfile)[0]):
        os.makedirs(os.path.split(args.outfile)[0], exist_ok=True)
    print(args.outfile)
    with open(args.outfile, 'w') as f:
        f.write(log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pddl_match_file", type=str)
    parser.add_argument("outfile", type=str)
    parser.add_argument("--print_acc", action='store_true')
    parser.add_argument("--print_subgoal", action='store_true')
    parser.add_argument("--print_sentence_match", action='store_true')
    parser.add_argument("--print_table", action='store_true') 
    parser.add_argument("--sentence_match_file", type=str) # ONLY NEEDED FOR PRINT SENTENCE MATCH
    parser.add_argument("--t2pData", type=str) # ONLY NEEDED FOR PRINT SENTENCE MATCH
    parser.add_argument("--split", choices=['all', 'seen', 'unseen'], default='all')
    parser.add_argument("--film_preprocess", action='store_true')
    args = parser.parse_args()
    main(args)