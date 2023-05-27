import os
import yaml
import json
import pprint
from model import Vanilla
from tqdm import tqdm
    
def main():
    # Load configures
    cfg = yaml.load(open('utils/config.yaml', 'r'), Loader=yaml.FullLoader)
    data_root = 'data/alfred_data/json_2.1.0'
    
    pp = pprint.PrettyPrinter(indent=4)
    model = Vanilla(cfg)
    for split in ['valid_seen', 'valid_unseen']:
        print('Split: %s'%split)
        for ep in tqdm(os.listdir(os.path.join(data_root, split)), leave=False):
            for trial in os.listdir(os.path.join(data_root, split, ep)):
                traj_data = json.load(open(os.path.join(data_root, split, ep, trial, 'traj_data.json'), 'r'))
                anns = traj_data['turk_annotations']['anns']
                for a_idx, ann in enumerate(anns):
                    plans, votes, probs, scores, text_plans = model.get_plan(ann['task_desc'])
                    
                    break
                break
            break
        break

if __name__ == "__main__":
    main()