from curses.ascii import SI
import json
from re import L
import sys
sys.path.append('../alfred')
sys.path.append('../alfred/gen')
from gen import constants

# all pickable objsb -- sliced excluded
OBJECTS_SET = constants.OBJECTS_SET
# for put action {parent:obj}
RECEPTACLE_MATCH = constants.VAL_RECEPTACLE_OBJECTS
ACTION_MATCH = constants.VAL_ACTION_OBJECTS
MRECEP_SET = constants.MOVABLE_RECEPTACLES_SET
RECEP_SET = constants.RECEPTACLES
SINGULAR = constants.OBJECTS_SINGULAR
PLURAL = constants.OBJECTS_PLURAL

def preprocess_objName(name):
    if name in PLURAL:
        name = SINGULAR[PLURAL.index(name)]
    if name not in OBJECTS_SET:
        for o in OBJECTS_SET:
            if o.lower() == name.lower():
                name = o
                break
    return name

def get_objList(taskType):
    # return avaliable objs and parent as {parent: obj set}
    objSet = OBJECTS_SET
    if 'slice' in taskType:
        objSet = objSet & ACTION_MATCH['Sliceable']
    if 'cool' in taskType:
        objSet = objSet & ACTION_MATCH['Coolable']
    if 'clean' in taskType:
        objSet = objSet & ACTION_MATCH['Cleanable']
    if 'heat' in taskType:
        objSet = objSet & ACTION_MATCH['Heatable']
    if 'look' in taskType:
        for o in objSet:
            if 'Sliced' in o:
                objSet.remove(o)
        return objSet
    parent = dict()
    for _parent, _oSet in RECEPTACLE_MATCH.items():
        _oSet = _oSet & objSet
        for o in _oSet:
            if 'Sliced' in o:
                _oSet.remove(o)
        parent[_parent] = _oSet
    if 'two' in taskType:
        for k in parent:
            oSet = parent[k]
            parent[k] = {PLURAL[SINGULAR.index(o.lower())] for o in oSet}
        return parent
    else:
        return parent
        

def main():
    # goal: taskParam
    generated = {}
    with open('data/template.json', 'r') as f:
        tplt = json.load(f)
    param = ['task_type', 'mrecep_target', 'object_sliced', 'object_target', 'parent_target', 'toggle_target']
    # check if not in alfred
    for taskType in tplt:
        if 'movable' in taskType:
            parent = get_objList(taskType)
            for m in MRECEP_SET:
                for p, oSet in parent.items():
                    # mrecep also should be puttable in parent
                    if m not in RECEPTACLE_MATCH[p]:
                        continue
                    for o in oSet:
                        if o not in RECEPTACLE_MATCH[m]:
                            continue
                        goals = tplt[taskType]
                        for goal in goals:
                            goal = goal.replace('[o]', o).replace('[p]', p).replace('[m]', m)
                            generated[goal] = dict(zip(param, [preprocess_objName(m), 'slice' in taskType, preprocess_objName(o), preprocess_objName(p), ""]))
        elif 'look' in taskType:
            objSet = get_objList(taskType)
            for lamp in ["DeskLamp", "FloorLamp"]:
                for o in objSet:
                    # remove lamp from o
                    if preprocess_objName(o) in ['FloorLamp', 'DeskLamp']:
                        continue
                    # remove unpickable object
                    # if preprocess_objName(o) in RECEP_SET-MRECEP_SET or preprocess_objName(o) in ['Television', 'LightSwitch', 'Window', 'Blinds', 'StoveKnob', 'Chair']:
                    #     continue
                    goals = tplt[taskType]
                    for goal in goals:
                        goal = goal.replace('[o]', o).replace('[t]', lamp)
                        generated[goal] = dict(zip(param, ["", 'slice' in taskType, preprocess_objName(o), "", preprocess_objName(lamp)]))
        else:
            parent = get_objList(taskType)
            for p, oSet in parent.items():
                for o in oSet:
                    goals = tplt[taskType]
                    for goal in goals:
                        goal = goal.replace('[o]', o).replace('[p]', p)
                        generated[goal] = dict(zip(param, ["", 'slice' in taskType, preprocess_objName(o), preprocess_objName(p), ""]))
    with open('data/fullGeneratedTask2Param.json', 'w') as f:
        json.dump(generated, f, indent=4)

if __name__ == "__main__":
    main()