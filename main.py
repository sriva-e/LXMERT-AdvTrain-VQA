import PIL.Image
import io
import json

import pandas as pd
import torch
import numpy as np
from processing_image import Preprocess
from visualizing_image import SingleImageViz
from modeling_frcnn import GeneralizedRCNN
from utils import Config
import utils
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
import pickle
import os

frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")

frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)

image_preprocess = Preprocess(frcnn_cfg)

lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")

VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
train_answers = pd.json_normalize(json.load(open("./data/annotations/v2_mscoco_train2014_annotations.json"))["annotations"])
train_questions = pd.json_normalize(json.load(open("./data/annotations/v2_OpenEnded_mscoco_train2014_questions.json"))["questions"])
train_images = pd.json_normalize(json.load(open("./data/annotations/captions_train2014.json"))["images"])

train_images["image_id"] = train_images["id"]
train_df = train_images.merge(train_questions, on='image_id').merge(train_answers, on='image_id')
train_df["answers"] = train_df["answers"].apply(lambda x: [y["answer"] for y in x])

predicted = []
vqa_answers = utils.get_data(VQA_URL)

for i, row in train_df.iterrows():
    # run lxmert
    URL = row["coco_url"]
    frcnn_visualizer = SingleImageViz(URL) # , id2obj=objids, id2attr=attrids)
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")
    test_question = row["question"]
    inputs = lxmert_tokenizer(
        test_question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    output_vqa = lxmert_vqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    # get prediction
    answers = set(row["answers"])
    pred_vqa = output_vqa["question_answering_score"].argmax(-1)
    ans = vqa_answers[pred_vqa]
    if ans in answers:
        predicted.append(1)
    else:
        predicted.append(0)

with open("./data/train_preds.json", "w") as f:
    f.write(json.dumps({"predictions": predicted}))