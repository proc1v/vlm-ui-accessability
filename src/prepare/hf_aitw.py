import os
import cv2
import re
import pdb
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from IPython.display import display
from PIL import Image, ImageDraw
from data_utils import is_english_simple, bbox_2_point


parent_dir = "/home/qinghong/data/GUI_database"
imgs_dir =  f"{parent_dir}/AITW/images"
anno_dir = f"{parent_dir}/AITW/metadata"

def draw_point_bbox(image_path, point=None, bbox=None, radius=5, line=3):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    if point is not None:
        x, y = point[0] * width, point[1] * height
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill='blue', outline='blue')
    if bbox is not None:
        x1, y1, x2, y2 = bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=line)

    image_draw = np.array(image)
    return image_draw

def data_transform(version='train', mini=False):
    aitw_data = json.load(open(f"{anno_dir}/aitw_data_{version}.json", 'r'))
    
    total_step = []
    step_i = 0
    for scenario in aitw_data:
        aitw_subset = aitw_data[scenario]
        for sample in aitw_subset:
            # print(sample)
            confirmed_task = sample[0]['goal']
    
            step_history = []
            for i, step in enumerate(sample):
                filename = step['img_filename']
                img_url = os.path.join(imgs_dir, filename) + '.png'
                if not os.path.exists(img_url):
                    print(img_url)
                    continue
                image = Image.open(img_url)
                action_id = step["action_type_id"]
                action_type = step["action_type_text"]
                # if action_id == 4:
                #     if action_type == "click":
                #         touch_point = step['touch']
                #         step_point = step['lift']
                #         click_point = [(touch_point[0] + lift_point[0]) / 2, (touch_point[1] + lift_point[1]) / 2]
                # elif action_type == 3:
                #     typed_text = step["type_text"]
                # print(step)
                total_step.append({
                                "split": version,
                                "id": "aitw_{}".format(step_i), 
                                # "annot_id": annot_id,
                                # "action_uid": step["action_uid"],
                                "domain": scenario,
                                "ep_id": step['ep_id'],
                                "step_id": i,
    
                                "task": confirmed_task,
                                "img_url": filename,
                                "img_size": image.size,
    
                                "action_type_id": action_id,
                                "action_type_text": action_type,
                                "annot_position": step['annot_position'],
                                "touch": step['touch'],
                                "lift": step['lift'],
                                "type_text": step['type_text'],
                                
                                "step": step,
                                # "step_repr": step_repr,
                                "step_history": step_history.copy(),
                                # "repr_history": repr_history.copy()
                                })
                # print(action_type)
                step_history.append(step)
                step_i += 1

                if mini and step_i > 50:
                    break
            if mini and step_i > 50:
                break

    return total_step

if __name__ == "__main__":
    for version in ['train', 'test', 'val']:
        data = data_transform(version=version)
        save_url = f"{anno_dir}/hf_{version}.json"
        with open(save_url, "w") as file:
            json.dump(data, file, indent=4)