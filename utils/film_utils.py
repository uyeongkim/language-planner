import json
import pickle
import os

TASK_TYPE_LIST = ['pick_cool_then_place_in_recep', 'pick_and_place_with_movable_recep', 'pick_and_place_simple', 'pick_two_obj_and_place',\
    'pick_heat_then_place_in_recep', 'look_at_obj_in_light', 'pick_clean_then_place_in_recep']

def film_task_type_label(task_type:str) -> int:
    film_dict = {
        'pick_and_place_simple': 2,
        'look_at_obj_in_light': 5,
        'pick_and_place_with_movable_recep': 1,
        'pick_two_obj_and_place': 3,
        'pick_clean_then_place_in_recep': 6,
        'pick_heat_then_place_in_recep': 4,
        'pick_cool_then_place_in_recep': 0}
    return film_dict[task_type]

def validation_result_to_film_pickle(json_file: str, save_folder: str, toggle=True) -> None:
    ###Convert our result file into film format.###
    pickle_root = os.path.dirname(os.path.abspath(__file__))
    if 'unseen' in json_file:
        pickle_file = pickle.load(open(os.path.join(pickle_root, 'instruction2_params_val_unseen_916_noappended.p'), 'rb'))
    else:
        pickle_file = pickle.load(open(os.path.join(pickle_root, 'instruction2_params_val_seen_916_noappended.p'), 'rb'))
    prediction = json.load(open(json_file, 'r'))
    
    def _film_preprocess(s):
        import string
        s = s.lower()
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        return s
    def _is_same_goal(our_goal, film_goal):
        return _film_preprocess(our_goal).strip() == film_goal.strip()
    def _pred_unique_param_config(goals):
        config = prediction[goals[0]]
        for g in goals[1:]:
            if prediction[g] != config:
                return False
        return True
    def _print_param_table(goal_in_prediction):
        different_key = set()
        for g in goal_in_prediction:
            if type(prediction[g]) == list:
                prediction[g] = prediction[g][0]
            for k in prediction[g]:
                
                if prediction[g][k] != prediction[goal_in_prediction[0]][k]:
                    different_key.add(k)
        for k in different_key:
            print("[%s]"%k)
            for g in goal_in_prediction:
                print("%30s | %s"%(g, prediction[g][k]))
    
    result = {}
    for goal in pickle_file:
        goal_in_prediction = [_g for _g in prediction if _is_same_goal(_g, goal)]
        if len(goal_in_prediction) == 0:
            print("WARNING: no prediction corresponing to [{}]".format(goal))
        if not _pred_unique_param_config(goal_in_prediction):
            print("WARNING: There are multiple parameter configuration predicted corresponding to one FILM goal")
            _print_param_table(goal_in_prediction)
        raw_pddl = prediction[goal_in_prediction[0]]
        pddl = {}
        if type(raw_pddl) == list:
            raw_pddl = raw_pddl[0]
        for k, v in raw_pddl.items():
            if k == 'object_sliced':
                pddl['sliced'] = 1 if v else 0
                continue
            if k == 'task_type':
                pddl['task_type'] = film_task_type_label(v)
                continue
            if v == '':
                pddl[k] = None
                continue
            if not toggle and k == 'toggle_target':
                continue
            pddl[k] = v
        result[goal] = pddl
    os.makedirs(save_folder, exist_ok=True)
    filenamme = os.path.split(json_file)[-1].replace('.json', '.p')
    pickle.dump(result, open(os.path.join(save_folder, filenamme), 'wb'))

def test_result_to_film_pickle(json_file:str, save_folder:str) -> None:
    ###Convert our result file into film format.###
    pickle_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if 'unseen' in json_file:
        pickle_file = pickle.load(open(os.path.join(pickle_root, 'data', 'film', 'instruction2_params_test_unseen_916_noappended.p'), 'rb'))
    else:
        pickle_file = pickle.load(open(os.path.join(pickle_root, 'data', 'film', 'instruction2_params_test_seen_916_noappended.p'), 'rb'))
    prediction = json.load(open(json_file, 'r'))
    
    def _film_preprocess(s):
        import string
        s = s.lower()
        s = ''.join(ch for ch in s if ch not in string.punctuation)
        return s
    def _is_same_goal(our_goal, film_goal):
        return _film_preprocess(our_goal).strip() == film_goal.strip()
    def _pred_unique_param_config(goals):
        config = prediction[goals[0]]
        for g in goals[1:]:
            if prediction[g] != config:
                return False
        return True
    def _print_param_table(goal_in_prediction):
        different_key = set()
        for g in goal_in_prediction:
            for k in prediction[g]:
                if prediction[g][k] != prediction[goal_in_prediction[0]][k]:
                    different_key.add(k)
        for k in different_key:
            print("[%s]"%k)
            for g in goal_in_prediction:
                print("%30s | %s"%(g, prediction[g][k]))
    
    result = {}
    for goal in pickle_file:
        goal_in_prediction = [_g for _g in prediction if _is_same_goal(_g, goal)]
        if len(goal_in_prediction) == 0:
            print("WARNING: no prediction corresponing to [{}]".format(goal))
        if not _pred_unique_param_config(goal_in_prediction):
            print("WARNING: There are multiple parameter configuration predicted corresponding to one FILM goal")
            _print_param_table(goal_in_prediction)
            
        raw_pddl = prediction[goal_in_prediction[0]]
        pddl = {}
        for k, v in raw_pddl.items():
            if k == 'object_sliced':
                pddl['sliced'] = 1 if v else 0
                continue
            if k == 'task_type':
                pddl['task_type'] = film_task_type_label(v)
                continue
            if v == '':
                pddl[k] = None
                continue
            pddl[k] = v
        result[goal] = pddl
    os.makedirs(save_folder, exist_ok=True)
    filenamme = os.path.split(json_file)[-1].replace('.json', '.p')
    pickle.dump(result, open(os.path.join(save_folder, filenamme), 'wb'))

def film_pickle_to_json(film_pickle:dict) -> dict:
    json_result = dict()
    for goal, pddl_param in film_pickle.items():
        replaced_pddl_param = dict()
        for key, val in pddl_param.items():
            if key == 'task_type':
                replaced_pddl_param[key] = TASK_TYPE_LIST[val]
            elif val == None:
                replaced_pddl_param[key] = ''
            elif key == 'sliced':
                replaced_pddl_param['object_sliced'] = (val == 1)
            else:
                replaced_pddl_param[key] = val
        json_result[goal] = replaced_pddl_param
    return json_result
