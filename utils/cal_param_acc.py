import argparse
import string
import os
import json
import numpy as np
from mdutils.mdutils import MdUtils

def film_preprocess(s):
    return ''.join([c for c in s.lower() if c not in set(string.punctuation)])

def find_most_correct_pddl(pddls, gtPddl):
    c = []
    for p in pddls:
        cnt = 0
        for k in gtPddl:
            if gtPddl[k] == p[k]:
                cnt += 1
        c.append(cnt)
    idx = np.argmax(c)
    return pddls[idx]

def convert_super(s):
    if s in ['Knife', 'ButterKnife']:
        return 'Knife'
    if s in ['FloorLamp', 'DeskLamp']:
        return 'FloorLamp'
    return s

def print_table(mdFile, gtPddl, pred):
    mdFile.new_line()
    rows = ["", "gt", "pred"]
    task_type_repr = {
        'look_at_obj_in_light': 'look',
        'pick_heat_then_place_in_recep': 'heat',
        'pick_cool_then_place_in_recep': 'cool',
        'pick_and_place_with_movable_recep': 'mrecep',
        'pick_two_obj_and_place': 'two',
        'pick_clean_then_place_in_recep': 'clean',
        'pick_and_place_simple': 'simple'
    }
    for k in ['task_type', 'mrecep_target', 'object_sliced', 'object_target', 'parent_target', 'toggle_target']:
        if k not in pred:
            rows.extend(["<span style=\"color: red\">%s</span>"%k, "<span style=\"color: red\">%s</span>"%gtPddl[k], "<span style=\"color: red\">NO PRED</span>"])
            continue
        if 'sliced' in k:
            if type(pred[k]) != type(True):
                rows.append("<span style=\"color: red\">%s</span>"%k)
                rows.append('<span style=\"color: red\">True</span>' if gtPddl[k] else '<span style=\"color: red\">False</span>')
                rows.append("<span style=\"color: red\">%s</span>"%pred[k])
            else:
                if gtPddl[k] != pred[k]:
                    rows.append("<span style=\"color: red\">%s</span>"%k)
                    rows.append('<span style=\"color: red\">True</span>' if gtPddl[k] else '<span style=\"color: red\">False</span>')
                    rows.append('<span style=\"color: red\">True</span>' if pred[k] else '<span style=\"color: red\">False</span>')
                else:
                    rows.extend([k, 'True' if gtPddl[k] else 'False', 'True' if pred[k] else 'False'])
        elif 'task_type' in k:
            if pred[k] not in task_type_repr:
                rows.append("<span style=\"color: red\">%s</span>"%k)
                rows.append('<span style=\"color: red\">%s</span>'%gtPddl[k])
                rows.append("<span style=\"color: red\">%s</span>"%pred[k])
            else:
                if gtPddl[k] != pred[k]:
                    rows.append("<span style=\"color: red\">%s</span>"%k)
                    rows.append('<span style=\"color: red\">%s</span>'%(task_type_repr[gtPddl[k]]))
                    rows.append('<span style=\"color: red\">%s</span>'%(task_type_repr[pred[k]]))
                else:
                    rows.extend([k, task_type_repr[gtPddl[k]], task_type_repr[pred[k]]])
        else:
            if gtPddl[k] != pred[k]:
                rows.append("<span style=\"color: red\">%s</span>"%k)
                rows.append('<span style=\"color: red\">%s</span>'%(gtPddl[k]))
                rows.append('<span style=\"color: red\">%s</span>'%(pred[k]))
            else:
                rows.extend([k, gtPddl[k], pred[k]])
    mdFile.new_table(columns=3, rows=7, text=rows, text_align='center')
    mdFile.new_line()

def print_acc(mdFile, all, type, m, o, s, p, t, cnt):
    mdFile.new_header(level=1, title='Parameter Prediction Accuracy')
    rows = ["", "# Correct", "# Total", "Accuracy"]
    rows.extend(['All', str(all), str(cnt), '%.2f'%(all/cnt*100)])
    rows.extend(['Type', str(type), str(cnt), '%.2f'%(type/cnt*100)])
    rows.extend(['mrecep', str(m), str(cnt), '%.2f'%(m/cnt*100)])
    rows.extend(['object', str(o), str(cnt), '%.2f'%(o/cnt*100)])
    rows.extend(['sliced', str(s), str(cnt), '%.2f'%(s/cnt*100)])
    rows.extend(['parent', str(p), str(cnt), '%.2f'%(p/cnt*100)])
    rows.extend(['toggle', str(t), str(cnt), '%.2f'%(t/cnt*100)])
    mdFile.new_table(columns=4, rows=8, text=rows, text_align='center')

def main(args):
    code_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # language-planner path
    data_path = os.path.join(code_path, 'data', 'alfred_data', 'json_2.1.0')
    with open(args.pddl_match_file, 'r') as f:
        prediction_pddl_match = json.load(f)
        if 'appended' in args.pddl_match_file:
            temp = {}
            for goal in prediction_pddl_match:
                temp[goal.split("[SEP]")[0]] = prediction_pddl_match[goal]
            prediction_pddl_match = temp
    if args.split == 'all':
        splits = ['valid_seen', 'valid_unseen']
    elif args.split == 'seen':
        splits = ['valid_seen']
    else:
        splits = ['valid_unseen']
        
    save_folder = os.path.join(code_path, 'param_log', args.split if 'appended' not in args.pddl_match_file else args.split+'Appended')
    save_file = os.path.split(args.pddl_match_file)[-1].replace('.json', '')+'-super' if args.super else os.path.split(args.pddl_match_file)[-1].replace('.json', '')
    os.makedirs(save_folder, exist_ok=True)
    print('Save log file in %s'%(os.path.join(save_folder, save_file)))
    mdFile = MdUtils(file_name=os.path.join(save_folder, save_file))

    all_correct = 0
    type_correct = 0
    m_correct = 0
    o_correct = 0
    s_correct = 0
    p_correct = 0
    t_correct = 0
    cnt = 0

    mdFile.new_header(level=1, title='PDDL parameter Table')
    for split in splits:
        for ep in os.listdir(os.path.join(data_path, split)):
            for trial in os.listdir(os.path.join(data_path, split, ep)):
                traj_data = json.load(open(os.path.join(data_path, split, ep, trial, 'traj_data.json'), 'r'))
                gtPddl = traj_data['pddl_params']
                if args.super:
                    gtPddl['toggle_target'] = convert_super(gtPddl['toggle_target'])
                gtPddl['task_type'] = traj_data['task_type']
                if 'film' in args.pddl_match_file:
                    tasks = [film_preprocess(ann['task_desc']) for ann in traj_data['turk_annotations']['anns']]
                else:
                    tasks = [ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]
                predicted_pddl = [prediction_pddl_match[t] for t in tasks]
                if 'film' in args.pddl_match_file:
                    temp = []
                    for pddls in predicted_pddl:
                        t = []
                        for pddl in pddls:
                            _p = pddl
                            _p['toggle_target'] = 'FloorLamp' if pddl['task_type'] == 'look_at_obj_in_light' else ''
                            t.append(_p)
                        temp.append(t)
                    predicted_pddl = temp
                if len([ann['task_desc'] for ann in traj_data['turk_annotations']['anns']]) != len(tasks) or len(tasks) != len(predicted_pddl):
                    raise Exception('No matching goal\ngtGoal: {}\nfoundGoal: {}'.format([ann['task_desc'] for ann in traj_data['turk_annotations']['anns']], tasks))
                for i, pddls in enumerate(predicted_pddl):
                    if args.super:
                        temp = []
                        for pddl in pddls:
                            pddl['object_target'] = convert_super(pddl['object_target'])
                            pddl['toggle_target'] = convert_super(pddl['toggle_target'])
                            temp.append(pddl)
                        pddls = temp
                    idx = np.where(np.array(pddls) == gtPddl, True, False)
                    cnt += 1
                    if True in idx:
                        all_correct += 1
                        type_correct += 1
                        m_correct += 1
                        o_correct += 1
                        s_correct += 1
                        p_correct += 1
                        t_correct += 1
                    else:
                        pddl = find_most_correct_pddl(pddls, gtPddl)
                        type_correct  = type_correct+1 if gtPddl['task_type'] == pddl['task_type'] else type_correct
                        m_correct  = m_correct+1 if gtPddl['mrecep_target'] == pddl['mrecep_target'] else m_correct
                        o_correct  = o_correct+1 if gtPddl['object_target'] == pddl['object_target'] else o_correct
                        s_correct  = s_correct+1 if gtPddl['object_sliced'] == pddl['object_sliced'] else s_correct
                        p_correct  = p_correct+1 if gtPddl['parent_target'] == pddl['parent_target'] else p_correct
                        t_correct  = t_correct+1 if gtPddl['toggle_target'] == pddl['toggle_target'] else t_correct
                        mdFile.write("**{}**\n".format(tasks[i].strip()))
                        print_table(mdFile, gtPddl, pddl)

    print_acc(mdFile, all_correct, type_correct, m_correct, o_correct, s_correct, p_correct, t_correct, cnt)
    mdFile.create_md_file()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pddl_match_file", type=str)
    parser.add_argument("--split", choices=['all', 'seen', 'unseen'], default='all')
    parser.add_argument("--super", action='store_true')
    args = parser.parse_args()
    main(args)