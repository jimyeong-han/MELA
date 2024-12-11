"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import logging
import copy

from torch.utils.data import DataLoader

from collections import OrderedDict

from lavis.datasets.datasets.multimodal_classification_datasets import (
    MultimodalClassificationDataset,
)
import random
from random import sample

class __DisplMixin:
    def displ_item(self, index):
        ann = self.annotation[index]
        vname = ann["video"]
        vpath = os.path.join(self.vis_root, vname)

        return OrderedDict(
            {"file": vpath, "question": ann["question"], "answer": ann["answer"]}
        )

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

ANS_MAPPING = {0:'A',1:'B',2:'C',3:'D',4:'E'}
# NextQA
class MCVideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def _load_auxiliary_mappings(self):
        pass
    
    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __getitem__(self, index):
        
        result = None
        while result is None:

            ann = self.annotation[index]
            qid = ann['qid'] 

            if 'QVHighlight' in qid:
                q = ann['query']
            else:
                q = ann['question']
            
            # set video clip if 'start'&'end' timestamp in data
            if 'start' in ann:
                start, end = float(ann['start']), float(ann['end'])
                clip = [start, end]
            else:
                clip = None       
            
            if 'VLEP' in qid:
                qa_prompt = 'Upon observing the provided frames, what is the most probable subsequent event?'
                events = 'Option A: ' + ann['a0'] + ' Option B: ' + ann['a1']
                qa_prompt = qa_prompt + ' ' + events
                loc_prompt = 'Does the information within the frame provide the necessary details to predict next event?'
                loc_prompt = qa_prompt + ' ' + loc_prompt
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1

            elif 'QVHighlight' in qid:
                duration = ann['duration']
                if 'relevant_windows' in ann: 
                    relevant_windows = ann['relevant_windows']
                else:
                    relevant_windows = None # for test
                pseudo_options = 'Option A: yes. Option B: no.'
                if q[-1] != '.':
                    q += '.'      
                loc_prompt = 'Question: ' + q +  ' ' + pseudo_options + ' Does the information within the frame provide the necessary details to accurately answer the given question?'
                qa_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'
                
                
            else:
                prompt = 'Question: ' + q
                for j in range(int(ann['num_option'])):
                    a = ann['a{}'.format(j)]
                    prompt += ' Option {}: '.format(ANS_MAPPING[j])
                    prompt += a
                hints = 'Options: ('
                #hints = 'Captions: ('
                for j in range(int(ann['num_option'])):
                    ans = ann['a{}'.format(str(j))]
                    hints += ans
                    hints += ' '
                hints += ')'
                qa_prompt = prompt + ' Considering the information presented in the frame, select the correct answer from the options.'
                loc_prompt = 'Question: ' + q +  ' ' + hints + ' Does the information within the frame provide the necessary details to accurately answer the given question?'                
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1
            
            try:
                if 'VLEP' in qid:
                    video_id = ann['video']
                    if ':' in video_id:
                        # we set absolute path for vlep as it takes multiple video source
                        # you may change below paths to you own path
                        video_path = '/nas-hdd/shoubin/vlep_ytb_clips_tars/videos/vlep_ytb_clips/'
                    else:
                        video_id = video_id[:-3]
                        video_path = '/nas-hdd/shoubin/videos/tvqa/videos_3fps_with_audio/'
                    vpath = os.path.join(video_path, video_id + '.mp4')
                else:
                    vpath = os.path.join(self.vis_root, str(ann['video']) + '.mp4')   
                    
                frms, indices, fps = self.vis_processor(vpath, clip_proposal=clip)
                frms = frms.permute(1, 0, 2, 3)
                assert len(frms) == self.vis_processor.n_frms
                
                if 'QVHighlight' in qid: 
                    time_stamp = [float(idx/fps) for idx in indices]
                    answers = []
                    if relevant_windows is not None:
                        for t in time_stamp:
                            flag = False
                            for span in relevant_windows:
                                if t >= float(span[0]) and t<= float(span[1]):
                                    answers.append('yes')
                                    flag = True 
                                    break
                            if not flag:
                                answers.append('no') 
                    else:
                        for t in time_stamp:
                            answers.append('no') # for test
                            
                    answers = '_'.join(answers)
                              
                result = True
            except Exception as e:
                
                print(f"Error while read file idx")
                print("video is: {}".format(ann['video']))
                index = random.randint(0, len(self.annotation) - 1)
                
        return {
            "video": frms,
            "qa_input": qa_prompt,
            "loc_input": loc_prompt,
            "qa_output": answers,
            "question_id": qid,
            'duration': duration
        }


class EventMCVideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.use_uniMD = True

        if self.use_uniMD:
            uniMD_oracle = False
            if uniMD_oracle:
                event_anno_file = "/data/VQA/data/star/STAR/txt_db/oracle_event_anno_uniMD.json"
            else:
                if self.data_type == 'train':
                    #event_anno_file = "/data/VQA/data/star/STAR/txt_db/events.json"
                    event_anno_file = "/data/VQA/data/star/STAR/txt_db/train_event_anno_uniMD_char2star.json"
                else:
                    #event_anno_file = "/data/VQA/data/star/STAR/txt_db/val_event_anno_uniMD_star_0.25.json"
                    event_anno_file = "/data/VQA/data/star/STAR/txt_db/val_event_anno_uniMD_0.4_char2star.json"
                    
        else:
            event_anno_file = "/data/VQA/data/star/STAR/txt_db/events.json"

        #print(event_anno_file)

        with open(event_anno_file, 'rb') as f:  # TODO: PATH
            self.event_anno = json.load(f)
        self.max_windows = 10  ## max events to use
        if self.data_type == 'train':
            obj_anno_file = "/data/VQA/data/star/STAR/txt_db/train_object_anno.json"
        else:
            obj_anno_file = "/data/VQA/data/star/STAR/txt_db/star_val_objects_YOLO_filtered.json"
        with open(obj_anno_file, 'rb') as f:  # TODO: PATH
            self.obj_anno = json.load(f)

        #### load GT events and build top-k triplets
        self.map_action()
        # text matcher
        self.sent_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1').cuda()
        for p in self.sent_model.parameters():
            p.requires_grad = False
        self.act_embs = self.sent_model.encode(self.act_list, show_progress_bar=False)

        output_trip_samples = True
        if output_trip_samples:
            out_json = []
            
        ##############################
        ######  Hyperparameters ######
        ##############################
        
        self.event_only = False # event 있는 애들만 실험
        self.events_in_prompt = True # events list로 prompt에 직접 넣어서
        self.events_in__loc_prompt = False
        self.use_cache = True
        self.use_glance = False
        self.except_fea = False
        self.use_obj = False # object 정보 활용
        self.charades_to_star = True
        self.use_topk_event_on_ans = False
        
        ##############################
        ######  Hyperparameters ######
        ##############################
        
        #if self.charades_to_star:
        if False:
            file_ptr = open("/data/VQA/data/star/STAR/txt_db/action_mapping_charades_to_star.txt", 'r')
            mapper = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            self.c2s = {a.split(" ")[0]:a.split(" ")[1] for a in mapper}
            
        

        print(self.data_type+" dataset loading")
        data_split = self.data_type
        if not data_split == "train":
            data_split = "val"
            
        file_idx = "cache_dataset/star_idx_map_"+data_split+".json"
        if self.use_uniMD:
            if self.data_type == 'train':
                file_topk = "cache_dataset/star_topk_events_uniMD_0.4_char2star_train.json"
                #file_topk = "cache_dataset/star_topk_events_"+data_split+".json"
            else:
                file_topk = "cache_dataset/star_topk_events_uniMD_0.4_char2star_"+data_split+".json"
        else:
            file_topk = "cache_dataset/star_topk_events_"+data_split+".json"
        #print(file_topk)

        """
        if self.use_glance:
            file_idx = "cache_dataset/glance_star_idx_map_"+data_split+".json"
            file_topk = "cache_dataset/glance_star_topk_events_"+data_split+".json"
            
            file_glanced = "cache_dataset/glance_star_events_"+data_split+".json"
            
            if self.use_cache and os.path.isfile(file_glanced):
                with open(file_glanced, 'rb') as f:
                    json_dump = json.load(f)
                    self.event_anno = json.loads(json_dump)
            else:
                self.glance_args = Args_sup()
        
                if data_split == 'train':
                    glance_dataset = VideoQADataset(self.glance_args.task_type, self.glance_args.train_data_file_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.app_feat_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.str2num_file.format(self.glance_args.base_data_dir),
                                                self.glance_args.event_anno_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.action_mapping_file.format(self.glance_args.base_data_dir), self.glance_args.max_feats,
                                                num_queries=self.glance_args.num_queries, is_train=False, return_label=False)
                    glance_dataloader = DataLoader(glance_dataset, batch_size=self.glance_args.batch_size, shuffle=False,
                                                collate_fn=VideoQACollator(task_type=self.glance_args.task_type).collate_batch)
                else:
                    glance_dataset = VideoQADataset(self.glance_args.task_type, self.glance_args.val_data_file_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.app_feat_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.str2num_file.format(self.glance_args.base_data_dir),
                                                self.glance_args.event_anno_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.action_mapping_file.format(self.glance_args.base_data_dir), self.glance_args.max_feats,
                                                num_queries=self.glance_args.num_queries, is_train=False, return_label=False)
                    glance_dataloader = DataLoader(glance_dataset, batch_size=self.glance_args.batch_size, shuffle=False,
                                                collate_fn=VideoQACollator(task_type=self.glance_args.task_type).collate_batch)
                
                #self.glance_args.event_pred_dim = len(self.act_list) + 1
                transformer = build_transformer(self.glance_args)
                glance = GF(
                    transformer,
                    num_queries=self.glance_args.num_queries,
                    feature_dim=self.glance_args.feature_dim,
                    output_dim=self.glance_args.output_dim,
                    event_pred_dim=self.glance_args.event_pred_dim,
                    qa_dataset=self.glance_args.qa_dataset   
                ).cuda()
                self.glance_args.device = next(glance.parameters()).device
                self.glance_args.reload_model_path = '/data/VQA/Glance-Focus/merged_model_4000.tar'
                checkpoint = torch.load(self.glance_args.reload_model_path)
                glance.load_state_dict(checkpoint['model_state_dict'])
                #matcher = build_matcher(self.glance_args)
                
                self.map_qid_vid = {ann['qid']: ann['video'] for ann in self.annotation}
                
                self.prep_glance(glance, glance_dataloader, self.glance_args)
                
                with open(file_glanced, 'w') as f:
                    json_dump = json.dumps(self.event_anno, cls=NumpyEncoder)
                    json.dump(json_dump, f)
        """
        if self.use_cache and os.path.isfile(file_idx) and os.path.isfile(file_topk):
            #print('loading from cache')
            with open(file_idx, 'rb') as f:
                self.ann_idx_map = json.load(f)
            with open(file_topk, 'rb') as f:
                json_dump = json.load(f)
                self.topk_events_anno = json.loads(json_dump)
        else:
            #print('building cache')
            self.ann_idx_map = []
            self.topk_events_anno = {}

            yes_trip = 0
            no_trip = 0
            for ann_idx, ann in enumerate(tqdm(self.annotation)): 
                qid = ann['qid']
                vid = ann['video']
                q = ann['question']
                ## load events from question
                q_events = ann["q_events"]
                ## load video events
                event_anno = self.event_anno[qid]
                #event_anno = self.event_anno[vid]
                events = event_anno['actions']
                span_list, hoi_list = [], []
                for e in events:
                    span_list.append(e[-2:])
                    hoi_list.append(e[0])
                if len(events) > self.max_windows:
                    l = [i for i in range(len(events))]
                    sample_idx = sample(l, self.max_windows)
                    span_list = [span_list[idx] for idx in sample_idx]
                    hoi_list = [hoi_list[idx] for idx in sample_idx]
                    
                duration = event_anno['duration']
                span = self.get_span_labels(span_list, duration, 10)    ## normalize times (if use glance, already normalized)
                
                triplets = self.build_triplet(span_list, hoi_list, q_events)
                if len(triplets) > 0:
                    topk_triplet, topk_indices = self.filter_topk_triplet(triplets, q, k=3)
                    if output_trip_samples:
                        out_dict = {
                            "qid": qid,
                            "question": q,
                            "q_events": q_events,
                            "answer": ann['a{}'.format(int(ann['answer']))],
                            "trip": triplets,
                            "t-k_trip": topk_triplet,
                            "t-k_idx": topk_indices,
                        }
                        out_json.append(out_dict)

                    topk_zip = '|'.join([','.join(a) for a in topk_triplet])
                    
                    topk_span = np.full((3,2,2), -1.)
                    for i, indices in enumerate(topk_indices):
                        for j, idx in enumerate(indices):
                            topk_span[i,j] = span[idx]*32

                    self.topk_events_anno[qid] = {
                        "t-k_trip": topk_zip,
                        "t-k_span": topk_span
                    }
                    yes_trip += 1
                    if self.event_only:
                        self.ann_idx_map.append(ann_idx)
                else:
                    self.topk_events_anno[qid] = {
                        "t-k_trip": "",
                        "t-k_span": np.full((3,2,2), -1.)
                    }
                    no_trip += 1
            logging.info("{} samples has triplets".format(yes_trip))
            logging.info("{} samples don't have triplets".format(no_trip))

            if output_trip_samples:
                with open("/data/VQA/SeViLA/topk_samples_w_ans.json", "w") as f_json:
                    json.dump(out_json, f_json)

            with open(file_idx, 'w') as f:
                json.dump(self.ann_idx_map, f)
            with open(file_topk, 'w') as f:
                json_dump = json.dumps(self.topk_events_anno, cls=NumpyEncoder)
                json.dump(json_dump, f)
            
        print(self.data_type+" dataset loaded")
        
    def merge_actions(self, actions):
        new_actions = []
        empty = True
        for act in actions:
            if empty:
                new_actions.append(act)
                empty = False
            else:
                na = copy.deepcopy(new_actions)
                flag = False
                for i, new_act in enumerate(na):
                    if act[0] == new_act[0]:
                        if act[1] == new_act[1] and act[2] == new_act[2]:
                            flag = True
                            break
                        if act[1] >= new_act[1]:
                            if act[1] >= new_act[2]:
                                pass
                            elif act[2] >= new_act[2]:
                                new_actions[i] = [act[0], new_act[1], act[2]]
                                flag = True
                                break
                            else:
                                new_actions[i] = [act[0], new_act[1], new_act[2]]
                                flag = True
                                break
                        elif new_act[1] <= act[2]:
                            if act[2] >= new_act[2]:
                                new_actions[i] = [act[0], act[1], act[2]]
                                flag = True
                                break
                            else:
                                new_actions[i] = [act[0], act[1], new_act[2]]
                                flag = True
                                break
                if not flag:
                    new_actions.append(act)
        if len(actions) > len(new_actions):
            return self.merge_actions(new_actions)
        else:
            return new_actions

    def map_action(self):
        #if self.use_uniMD and not self.charades_to_star:
        if False:
            file_ptr = open("/data/VQA/data/star/STAR/txt_db/action_mapping_charades.txt", 'r')
            #file_ptr = open("/data/VQA/data/star/STAR/txt_db/action_mapping.txt", 'r')
        else:
            file_ptr = open("/data/VQA/data/star/STAR/txt_db/action_mapping.txt", 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.act_list = [a.split(" ", 1)[1] for a in actions]
    
        self.act2key = {action:key for key, action in enumerate(self.act_list)}
        self.key2act = {key:action for key, action in enumerate(self.act_list)}

        return

    def build_triplet(self, span, hoi, q_events):
        """
        vid_events = [span, hoi] : (time, label)
        
        span : time
        hoi : label(key)
        """
        mapped_q_events = [] ## [order in hoi, sent]
        triplets = []
        
        hoi_str = [self.key2act[key] for key in hoi if key >= 0]  ## transform video event keys to string
        
        if len(hoi_str) == 0: ## GT event도 없을 때
            return triplets
        
        event_pair = set()
        for q_e in q_events:
            doc_emb = self.sent_model.encode(hoi_str, show_progress_bar=False)
            query_emb = self.sent_model.encode(q_e, show_progress_bar=False)
            scores = util.dot_score(query_emb, doc_emb)[0].cpu() ## map question event to video event
            event_idx = scores.argmax().item() ## mapped index in hoi
            event_key = hoi[event_idx] ## event key
            event_str = self.key2act[event_key]
            mapped_q_events.append([event_idx, event_str])
            
            event_span = span[event_idx]
            for piv_idx, s in enumerate(span):
                piv_key = hoi[piv_idx]
                if piv_key < 0: ## handle padding
                    break
                piv_str = self.key2act[piv_key]
                
                if event_idx == piv_idx: ## 자신 제외
                    continue
                if event_span[0] < s[0]:
                    if event_span[1] < s[1]:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([event_key, "before", piv_key, event_str+" before "+piv_str, event_idx, piv_idx])
                            event_pair.add((event_key, piv_key))
                    else:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([piv_key, "while", event_key, piv_str+" while "+event_str, piv_idx, event_idx])
                            event_pair.add((piv_key, event_key))
                else:
                    if event_span[1] > s[1]:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([event_key, "after", piv_key, event_str+" after "+piv_str, event_idx, piv_idx])
                            event_pair.add((event_key, piv_key))
                    else:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([event_key, "while", piv_key, event_str+" while "+piv_str, event_idx, piv_idx])
                            event_pair.add((event_key, piv_key))
                         
        return triplets    
        
    def filter_topk_triplet(self, triplets, q_sent, k):
        
        ## 기존 prompt에서 question 추출
        triplets = np.array(triplets)

        # index들 추출
        triplet_indices = triplets[:,4:].astype(int)
        

        top_k_triplet = []
        ## Todo: 질문 - triplet-q_sent 간의 attention 등 계산해 top-k 선택   
        ## opt 1. similarity 계산
        triplet_sents = triplets[:,-3]
        
        triplet_emb = self.sent_model.encode(list(triplet_sents), show_progress_bar=False)
        query_emb = self.sent_model.encode(q_sent, show_progress_bar=False)
        
        scores = util.dot_score(query_emb, triplet_emb)[0].cpu()
        topk_idx = scores.argsort()[-k:].flip(dims=(0,)).tolist()
        
        top_k_triplet = triplets[topk_idx,:-3]
        top_k_indices = triplet_indices[topk_idx, :]
        top_k_indices = [list(a) for a in top_k_indices] 
        
        return [list(a) for a in top_k_triplet], [[int(a) for a in list(b)]for b in top_k_indices] 

    def collect_events(self, triplets):
        events = set()

        for triplet in triplets:
            events.add(triplet[0])
            events.add(triplet[2])
        events = list(events)
        
        final_events = []
        for e in events:
            final_events.append([int(e)])
        
        return final_events

    def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
        rows = []
        for a in aa:
            rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
        return np.concatenate(rows, axis=0).reshape(-1, fixed_length)
    
    def get_span_labels(self, windows, ctx_l, max_pad=10):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) == 0:
            windows = [[]]
        windows = np.pad(windows, ((0, max_pad), (0, 2)), 'constant', constant_values=-1)[:max_pad,:2]
        windows = torch.Tensor(windows) / ctx_l  # normalized windows in xx
        return windows

    def _load_auxiliary_mappings(self):
        pass
    
    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __len__(self):
        if self.event_only:
            return len(self.ann_idx_map)
        else:
            return len(self.annotation)

    def __getitem__(self, index):
        if self.event_only:
            index = self.ann_idx_map[index]


        result = None
        while result is None:

            ann = self.annotation[index]
            qid = ann['qid'] 
            
            ### preprocess top-k on dataset (str 형태로 전달)
            topk = self.topk_events_anno[qid]

            if 'QVHighlight' in qid:
                q = ann['query']
            else:
                q = ann['question']
            
            # set video clip if 'start'&'end' timestamp in data
            if 'start' in ann:
                start, end = float(ann['start']), float(ann['end'])
                clip = [start, end]
            else:
                clip = None       
            
            if 'VLEP' in qid:
                qa_prompt = 'Upon observing the provided frames, what is the most probable subsequent event?'
                events = 'Option A: ' + ann['a0'] + ' Option B: ' + ann['a1']
                qa_prompt = qa_prompt + ' ' + events
                loc_prompt = 'Does the information within the frame provide the necessary details to predict next event?'
                loc_prompt = qa_prompt + ' ' + loc_prompt
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1

            elif 'QVHighlight' in qid:
                duration = ann['duration']
                if 'relevant_windows' in ann: 
                    relevant_windows = ann['relevant_windows']
                else:
                    relevant_windows = None # for test
                pseudo_options = 'Option A: yes. Option B: no.'
                if q[-1] != '.':
                    q += '.'      
                loc_prompt = 'Question: ' + q +  ' ' + pseudo_options + ' Does the information within the frame provide the necessary details to accurately answer the given question?'
                qa_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'
                
            else:
                prompt = 'Question: ' + q
                for j in range(int(ann['num_option'])):
                    a = ann['a{}'.format(j)]
                    prompt += ' Option {}: '.format(ANS_MAPPING[j])
                    prompt += a
                hints = 'Options: ('
                #hints = 'Captions: ('
                for j in range(int(ann['num_option'])):
                    ans = ann['a{}'.format(str(j))]
                    hints += ans
                    hints += ' '
                hints += ')'

                vid = ann['video']
                use_obj = self.use_obj
                if vid not in self.obj_anno.keys():
                    use_obj = False
                if len(self.obj_anno[vid]) == 0:
                    use_obj = False
                
                if use_obj:
                    obj_prompt = " Objects: ("
                    if vid in self.obj_anno.keys():
                        for j in self.obj_anno[vid]:
                            obj_prompt += j
                            obj_prompt += ", "
                        obj_prompt = obj_prompt[:-2] + ")"
                    else:
                        obj_prompt = obj_prompt[:-1] + "None"
                    prompt += obj_prompt

                if self.use_topk_event_on_ans:
                    topk_triplets = topk["t-k_trip"]
                    topk_triplet = [aaa.split(',') for aaa in topk_triplets.split("|")]
                    if len(topk_triplet[0][0]) > 0:
                        event_anno = self.collect_events(topk_triplet)  
                    else:
                        event_anno = []                      
                else:
                    event_anno = self.event_anno[ann['qid']]["actions"]
                
                flag_pass_event = False
                if len(event_anno) == 0:
                    flag_pass_event = True
                if self.except_fea and qid.startswith("Feasibility"):
                    flag_pass_event = True
                
                if self.events_in_prompt and not flag_pass_event:
                # if self.events_in_prompt:    
                    #event_anno = self.event_anno[ann['qid']]["actions"]
                    
                    #if len(event_anno) == 0:
                    #    pass
                    #elif len(event_anno) > 4:
                    #    event_prompt = "Events: ("
                    #    #l = [i for i in range(len(event_anno))]
                    #    #sample_idx = sample(l, 6)
                    #    l = [i for i in range(4)]
                    #    sample_idx = l
                    #    for j in sample_idx:
                    #        event_prompt += self.key2act[event_anno[j][0]]
                    #        event_prompt += ", "
                    #    event_prompt = event_prompt[:-2] + ")"
                    #    prompt += event_prompt
                    #else:
                    event_prompt = " Actions in video: ("
                    for j in event_anno:
                        event_prompt += self.key2act[j[0]]
                        event_prompt += ", "
                    event_prompt = event_prompt[:-2] + ")"
                    prompt += event_prompt
                    if use_obj:
                        qa_prompt = prompt + ' Considering the information presented in the frame and list of objects and actions, select the correct answer from the options.'
                    else:
                        qa_prompt = prompt + ' Considering the information presented in the frame and list of actions, select the correct answer from the options.'
                else:
                    if use_obj:
                        qa_prompt = prompt + ' Considering the information presented in the frame and list of objects, select the correct answer from the options.'
                    else:
                        qa_prompt = prompt + ' Considering the information presented in the frame, select the correct answer from the options.'
                #qa_prompt = prompt
                
                if self.events_in__loc_prompt and len(event_anno) > 0:
                    loc_prompt = event_prompt + ' Question: ' + q +  ' ' + hints + ' Does the information within the frame provide the necessary details to accurately answer the given question?'  
                    #loc_prompt = event_prompt + ' Question: ' + q +  ' ' + hints + 'Considering given list of actions in video, does the information within the frame provide the necessary details to accurately answer the given question?'                
                else:
                    loc_prompt = 'Question: ' + q +  ' ' + hints + ' Does the information within the frame provide the necessary details to accurately answer the given question?'                
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1
            
            try:
                if 'VLEP' in qid:
                    video_id = ann['video']
                    if ':' in video_id:
                        # we set absolute path for vlep as it takes multiple video source
                        # you may change below paths to you own path
                        video_path = '/nas-hdd/shoubin/vlep_ytb_clips_tars/videos/vlep_ytb_clips/'
                    else:
                        video_id = video_id[:-3]
                        video_path = '/nas-hdd/shoubin/videos/tvqa/videos_3fps_with_audio/'
                    vpath = os.path.join(video_path, video_id + '.mp4')
                else:
                    vpath = os.path.join(self.vis_root, str(ann['video']) + '.mp4')   
                    
                frms, indices, fps = self.vis_processor(vpath, clip_proposal=clip) # vision feature extraction
                frms = frms.permute(1, 0, 2, 3)
                assert len(frms) == self.vis_processor.n_frms # 32 default
                
                if 'QVHighlight' in qid: 
                    time_stamp = [float(idx/fps) for idx in indices]
                    answers = []
                    if relevant_windows is not None:
                        for t in time_stamp:
                            flag = False
                            for span in relevant_windows:
                                if t >= float(span[0]) and t<= float(span[1]):
                                    answers.append('yes')
                                    flag = True 
                                    break
                            if not flag:
                                answers.append('no') 
                    else:
                        for t in time_stamp:
                            answers.append('no') # for test
                            
                    answers = '_'.join(answers)
                              
                result = True
            except Exception as e:
                
                print(f"Error while read file idx")
                print("video is: {}".format(ann['video']))
                index = random.randint(0, len(self.annotation) - 1)
            

            """
            ## event part added
            
            q_events = " | ".join(ann["q_events"])    
            
            ## load GT events
            event_anno = self.event_anno[qid]
            duration = event_anno['duration']
            events = event_anno['actions']
            span_list, hoi_list = [], []
            for e in events:
                span_list.append(e[-2:])
                hoi_list.append(e[0])
            if len(events) > self.max_windows:
                l = [i for i in range(len(events))]
                sample_idx = sample(l, self.max_windows)
                span_list = [span_list[idx] for idx in sample_idx]
                hoi_list = [hoi_list[idx] for idx in sample_idx]
            span = self.get_span_labels(span_list, duration, 10)    ## normalize times
            
            hoi_list = np.pad(hoi_list, (0, 10), 'constant', constant_values=-1)[:10]
        
            hoi = torch.Tensor(hoi_list).long()
            
            duration = 1
            """

                
        return {
            "video": frms,
            "qa_input": qa_prompt,
            "loc_input": loc_prompt,
            "qa_output": answers,
            "question_id": qid,
            'duration': duration,
            #"q_events": q_events,
            #"span": span,
            #"hoi": hoi
            "topk_triplets": topk["t-k_trip"],
            "topk_span": np.array(topk["t-k_span"])
        }

    def collater(self, samples):
    #     q_events = []
    #     span = []
    #     hoi = []

    #     for i in range(len(samples)):
    #         span.append(samples[i].pop("span"))
    #         hoi.append(samples[i].pop("hoi"))

    #     samples = default_collate(samples)

    #     samples["q_events"] = q_events
    #     samples['span'] = span
    #     samples['hoi'] = hoi
        #print(samples)
        return default_collate(samples)
    
  
class EventOEVideoQADataset(MultimodalClassificationDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        event_anno_file = "/data/VQA/data/egotaskqa/data/metadata/video_events.json"
        with open(event_anno_file, 'rb') as f:  # TODO: PATH
            self.event_anno = json.load(f)
        self.max_windows = 10  ## max events to use

        #### load GT events and build top-k triplets
        self.map_action()
        # text matcher
        self.sent_model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-cos-v1').cuda()
        for p in self.sent_model.parameters():
            p.requires_grad = False
        self.act_embs = self.sent_model.encode(self.act_list, show_progress_bar=False)

        output_trip_samples = True
        if output_trip_samples:
            out_json = []

        self.event_only = False # event 있는 애들만 실험
        self.events_in_prompt = False # events list로 prompt에 직접 넣어서
        self.use_cache = True

        data_split = ann_paths[0].split('/')[-1].split('.')[0]
        print(data_split+" dataset loading")
            
        file_idx = "cache_dataset/egoqa/merged_idx_map_"+data_split+".json"
        file_topk = "cache_dataset/egoqa/merged_topk_events_"+data_split+".json"

        self.use_glance = False
        """
        if self.use_glance:
            file_idx = "cache_dataset/glance_merged_agqa_idx_map_"+data_split+".json"
            file_topk = "cache_dataset/glance_merged_agqa_topk_events_"+data_split+".json"
            
            file_glanced = "cache_dataset/glance_merged_agqa_events_"+data_split+".json"
            
            if self.use_cache and os.path.isfile(file_glanced):
                with open(file_glanced, 'rb') as f:
                    json_dump = json.load(f)
                    self.event_anno = json.loads(json_dump)
            else:
                self.glance_args = Args_sup()
        
                if data_split == 'train':
                    glance_dataset = VideoQADataset(self.glance_args.task_type, self.glance_args.train_data_file_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.app_feat_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.str2num_file.format(self.glance_args.base_data_dir),
                                                self.glance_args.event_anno_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.action_mapping_file.format(self.glance_args.base_data_dir), self.glance_args.max_feats,
                                                num_queries=self.glance_args.num_queries, is_train=False, return_label=False)
                    glance_dataloader = DataLoader(glance_dataset, batch_size=self.glance_args.batch_size, shuffle=False,
                                                collate_fn=VideoQACollator(task_type=self.glance_args.task_type).collate_batch)
                else:
                    glance_dataset = VideoQADataset(self.glance_args.task_type, self.glance_args.val_data_file_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.app_feat_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.str2num_file.format(self.glance_args.base_data_dir),
                                                self.glance_args.event_anno_path.format(self.glance_args.base_data_dir),
                                                self.glance_args.action_mapping_file.format(self.glance_args.base_data_dir), self.glance_args.max_feats,
                                                num_queries=self.glance_args.num_queries, is_train=False, return_label=False)
                    glance_dataloader = DataLoader(glance_dataset, batch_size=self.glance_args.batch_size, shuffle=False,
                                                collate_fn=VideoQACollator(task_type=self.glance_args.task_type).collate_batch)
                
                #self.glance_args.event_pred_dim = len(self.act_list) + 1
                transformer = build_transformer(self.glance_args)
                glance = GF(
                    transformer,
                    num_queries=self.glance_args.num_queries,
                    feature_dim=self.glance_args.feature_dim,
                    output_dim=self.glance_args.output_dim,
                    event_pred_dim=self.glance_args.event_pred_dim,
                    qa_dataset=self.glance_args.qa_dataset   
                ).cuda()
                self.glance_args.device = next(glance.parameters()).device
                self.glance_args.reload_model_path = '/data/VQA/Glance-Focus/merged_model_4000.tar'
                checkpoint = torch.load(self.glance_args.reload_model_path)
                glance.load_state_dict(checkpoint['model_state_dict'])
                #matcher = build_matcher(self.glance_args)
                
                self.map_qid_vid = {ann['qid']: ann['video'] for ann in self.annotation}
                
                self.prep_glance(glance, glance_dataloader, self.glance_args)
                
                with open(file_glanced, 'w') as f:
                    json_dump = json.dumps(self.event_anno, cls=NumpyEncoder)
                    json.dump(json_dump, f)
        """
        if self.use_cache and os.path.isfile(file_idx) and os.path.isfile(file_topk):
            with open(file_idx, 'rb') as f:
                self.ann_idx_map = json.load(f)
            with open(file_topk, 'rb') as f:
                json_dump = json.load(f)
                self.topk_events_anno = json.loads(json_dump)
        else:
            self.ann_idx_map = []
            self.topk_events_anno = {}

            yes_trip = 0
            no_trip = 0
            for ann_idx, ann in enumerate(tqdm(self.annotation)): 
                qid = ann['qid']
                vid = ann['video']
                q = ann['question']
                ## load events from question
                q_events = ann["q_events"]
                ## load GT events
                #event_anno = self.event_anno[qid]
                event_anno = self.event_anno[vid]
                events = event_anno['actions']
                span_list, hoi_list = [], []
                for e in events:
                    span_list.append(e[-2:])
                    hoi_list.append(e[0])
                if len(events) > self.max_windows:
                    l = [i for i in range(len(events))]
                    sample_idx = sample(l, self.max_windows)
                    span_list = [span_list[idx] for idx in sample_idx]
                    hoi_list = [hoi_list[idx] for idx in sample_idx]
                    
                duration = event_anno['duration']
                span = self.get_span_labels(span_list, duration, 10)    ## normalize times (if use glance, already normalized)
                
                triplets = self.build_triplet(span_list, hoi_list, q_events)
                if len(triplets) > 0:
                    topk_triplet, topk_indices = self.filter_topk_triplet(triplets, q, k=3)
                    if output_trip_samples:
                        out_dict = {
                            "qid": qid,
                            "question": q,
                            "q_events": q_events,
                            "answer": ann['a{}'.format(int(ann['answer']))],
                            "trip": triplets,
                            "t-k_trip": topk_triplet,
                            "t-k_idx": topk_indices,
                        }
                        out_json.append(out_dict)

                    topk_zip = '|'.join([','.join(a) for a in topk_triplet])
                    
                    topk_span = np.full((3,2,2), -1.)
                    for i, indices in enumerate(topk_indices):
                        for j, idx in enumerate(indices):
                            topk_span[i,j] = span[idx]*32

                    self.topk_events_anno[qid] = {
                        "t-k_trip": topk_zip,
                        "t-k_span": topk_span
                    }
                    yes_trip += 1
                    if self.event_only:
                        self.ann_idx_map.append(ann_idx)
                else:
                    self.topk_events_anno[qid] = {
                        "t-k_trip": "",
                        "t-k_span": np.full((3,2,2), -1.)
                    }
                    no_trip += 1
            logging.info("{} samples has triplets".format(yes_trip))
            logging.info("{} samples don't have triplets".format(no_trip))

            if output_trip_samples:
                with open("/data/VQA/SeViLA/mid_data/egoqa/topk_samples_w_ans.json", "w") as f_json:
                    json.dump(out_json, f_json)

            with open(file_idx, 'w') as f:
                json.dump(self.ann_idx_map, f)
            with open(file_topk, 'w') as f:
                json_dump = json.dumps(self.topk_events_anno, cls=NumpyEncoder)
                json.dump(json_dump, f)
            
        print(data_split+" dataset loaded")
    """  
    def prep_glance(self, glance, glance_dataloader, glance_args):
        glance.eval()
        device = glance_args.device
        pbar = tqdm(total=len(glance_dataloader))
        
        new_event_anno = {}
        print("Glancing...")
        with torch.no_grad():
            for b, batch in enumerate(glance_dataloader):
                
                frame_features = torch.stack(batch['visual_inputs']).to(device)
                visual_attention_mask = torch.ones(frame_features.shape[:-1], dtype=torch.float).to(device)
                
                memory_cache = glance(frame_features, visual_attention_mask, None, encode_and_save=True, glance=True)
                outputs_event = glance(frame_features, visual_attention_mask, None, encode_and_save=False, glance=True,
                                    memory_cache=memory_cache, query_type='event')
                
                pred_hois = outputs_event['pred_logits'].argmax(dim=2) # B, 10
                
                qids = batch['question_ids']
                for i, qid in enumerate(qids):
                    vid = self.map_qid_vid[qid]
                    
                    duration = self.event_anno[vid]['duration']
                    prev_actions = self.event_anno[vid]['actions']
                    # if len(prev_actions) > 0:
                    #     new_event_anno[qid] ={
                    #         "duration": duration,
                    #         "actions": prev_actions
                    #         }
                    #     continue

                    hois = pred_hois[i]
                    spans = span_cxw_to_xx(outputs_event['pred_spans'][i]) * duration
                    
                    actions = []
                    
                    for j, hoi in enumerate(hois):
                        actions.append([hoi.item(), spans[j][0].item(), spans[j][1].item()])
                        
                    actions = self.merge_actions(actions)
                    
                    new_event_anno[vid] ={
                        "duration": duration,
                        "actions": actions
                        }
                    
                pbar.update(1)
        self.event_anno = new_event_anno
        return
    """
    def merge_actions(self, actions):
        new_actions = []
        empty = True
        for act in actions:
            if empty:
                new_actions.append(act)
                empty = False
            else:
                na = copy.deepcopy(new_actions)
                flag = False
                for i, new_act in enumerate(na):
                    if act[0] == new_act[0]:
                        if act[1] == new_act[1] and act[2] == new_act[2]:
                            flag = True
                            break
                        if act[1] >= new_act[1]:
                            if act[1] >= new_act[2]:
                                pass
                            elif act[2] >= new_act[2]:
                                new_actions[i] = [act[0], new_act[1], act[2]]
                                flag = True
                                break
                            else:
                                new_actions[i] = [act[0], new_act[1], new_act[2]]
                                flag = True
                                break
                        elif new_act[1] <= act[2]:
                            if act[2] >= new_act[2]:
                                new_actions[i] = [act[0], act[1], act[2]]
                                flag = True
                                break
                            else:
                                new_actions[i] = [act[0], act[1], new_act[2]]
                                flag = True
                                break
                if not flag:
                    new_actions.append(act)
        if len(actions) > len(new_actions):
            return self.merge_actions(new_actions)
        else:
            return new_actions

    def map_action(self):
        with open("/data/VQA/data/egotaskqa/data/metadata/hois.json", 'r') as f:
            actions = json.load(f)
            
        self.act_list = [a.replace("[", "").replace("]", "").replace(",", "/") for a in actions]
    
        self.act2key = {action:key for key, action in enumerate(self.act_list)}
        self.key2act = {key:action for key, action in enumerate(self.act_list)}

        return

    def build_triplet(self, span, hoi, q_events):
        """
        vid_events = [span, hoi] : (time, label)
        
        span : time
        hoi : label(key)
        """
        mapped_q_events = [] ## [order in hoi, sent]
        triplets = []
        
        hoi_str = [self.key2act[key] for key in hoi if key >= 0]  ## transform video event keys to string
        
        if len(hoi_str) == 0: ## GT event도 없을 때
            return triplets
        
        event_pair = set()
        for q_e in q_events:
            doc_emb = self.sent_model.encode(hoi_str, show_progress_bar=False)
            query_emb = self.sent_model.encode(q_e, show_progress_bar=False)
            scores = util.dot_score(query_emb, doc_emb)[0].cpu() ## map question event to video event
            event_idx = scores.argmax().item() ## mapped index in hoi
            event_key = hoi[event_idx] ## event key
            event_str = self.key2act[event_key]
            mapped_q_events.append([event_idx, event_str])
            
            event_span = span[event_idx]
            for piv_idx, s in enumerate(span):
                piv_key = hoi[piv_idx]
                if piv_key < 0: ## handle padding
                    break
                piv_str = self.key2act[piv_key]
                
                if event_idx == piv_idx: ## 자신 제외
                    continue
                if event_span[0] < s[0]:
                    if event_span[1] < s[1]:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([event_key, "before", piv_key, event_str+" before "+piv_str, event_idx, piv_idx])
                            event_pair.add((event_key, piv_key))
                    else:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([piv_key, "while", event_key, piv_str+" while "+event_str, piv_idx, event_idx])
                            event_pair.add((piv_key, event_key))
                else:
                    if event_span[1] > s[1]:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([event_key, "after", piv_key, event_str+" after "+piv_str, event_idx, piv_idx])
                            event_pair.add((event_key, piv_key))
                    else:
                        if (event_key, piv_key) not in event_pair and (piv_key, event_key) not in event_pair:
                            triplets.append([event_key, "while", piv_key, event_str+" while "+piv_str, event_idx, piv_idx])
                            event_pair.add((event_key, piv_key))
                         
        return triplets    
        
    def filter_topk_triplet(self, triplets, q_sent, k):
        
        ## 기존 prompt에서 question 추출
        triplets = np.array(triplets)

        # index들 추출
        triplet_indices = triplets[:,4:].astype(int)
        

        top_k_triplet = []
        ## Todo: 질문 - triplet-q_sent 간의 attention 등 계산해 top-k 선택   
        ## opt 1. similarity 계산
        triplet_sents = triplets[:,-3]
        
        triplet_emb = self.sent_model.encode(list(triplet_sents), show_progress_bar=False)
        query_emb = self.sent_model.encode(q_sent, show_progress_bar=False)
        
        scores = util.dot_score(query_emb, triplet_emb)[0].cpu()
        topk_idx = scores.argsort()[-k:].flip(dims=(0,)).tolist()
        
        top_k_triplet = triplets[topk_idx,:-3]
        top_k_indices = triplet_indices[topk_idx, :]
        top_k_indices = [list(a) for a in top_k_indices] 
        
        return [list(a) for a in top_k_triplet], [[int(a) for a in list(b)]for b in top_k_indices] 


    def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
        rows = []
        for a in aa:
            rows.append(np.pad(a, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length])
        return np.concatenate(rows, axis=0).reshape(-1, fixed_length)
    
    def get_span_labels(self, windows, ctx_l, max_pad=10):
        """
        windows: list([st, ed]) in seconds. E.g. [[26, 36]], corresponding st_ed clip_indices [[13, 17]] (inclusive)
            Note a maximum of `self.max_windows` windows are used.
        returns Tensor of shape (#windows, 2), each row is [center, width] normalized by video length
        """
        if len(windows) == 0:
            windows = [[]]
        windows = np.pad(windows, ((0, max_pad), (0, 2)), 'constant', constant_values=-1)[:max_pad,:2]
        windows = torch.Tensor(windows) / ctx_l  # normalized windows in xx
        return windows

    def _load_auxiliary_mappings(self):
        pass
    
    def _get_answer_label(self, answer):
        if answer in self.class_labels:
            return self.class_labels[answer]
        else:
            return len(self.class_labels)

    def __len__(self):
        if self.event_only:
            return len(self.ann_idx_map)
        else:
            return len(self.annotation)

    def __getitem__(self, index):
        if self.event_only:
            index = self.ann_idx_map[index]


        result = None
        while result is None:

            ann = self.annotation[index]
            qid = ann['qid']

            if 'QVHighlight' in qid:
                q = ann['query']
            else:
                q = ann['question']
            
            # set video clip if 'start'&'end' timestamp in data
            if 'start' in ann:
                start, end = float(ann['start']), float(ann['end'])
                clip = [start, end]
            else:
                clip = None       
            
            if 'VLEP' in qid:
                qa_prompt = 'Upon observing the provided frames, what is the most probable subsequent event?'
                events = 'Option A: ' + ann['a0'] + ' Option B: ' + ann['a1']
                qa_prompt = qa_prompt + ' ' + events
                loc_prompt = 'Does the information within the frame provide the necessary details to predict next event?'
                loc_prompt = qa_prompt + ' ' + loc_prompt
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1

            elif 'QVHighlight' in qid:
                duration = ann['duration']
                if 'relevant_windows' in ann: 
                    relevant_windows = ann['relevant_windows']
                else:
                    relevant_windows = None # for test
                pseudo_options = 'Option A: yes. Option B: no.'
                if q[-1] != '.':
                    q += '.'      
                loc_prompt = 'Question: ' + q +  ' ' + pseudo_options + ' Does the information within the frame provide the necessary details to accurately answer the given question?'
                qa_prompt = 'Considering the information presented in the frame, select the correct answer from the options.'
                
            else:
                prompt = 'Question: ' + q
                for j in range(int(ann['num_option'])):
                    a = ann['a{}'.format(j)]
                    prompt += ' Option {}: '.format(ANS_MAPPING[j])
                    prompt += a
                hints = 'Options: ('
                #hints = 'Captions: ('
                for j in range(int(ann['num_option'])):
                    ans = ann['a{}'.format(str(j))]
                    hints += ans
                    hints += ' '
                hints += ')'
                
                if self.events_in_prompt:
                    event_anno = self.event_anno[ann['video']]["actions"]
                    if len(event_anno) == 0:
                        pass
                    elif len(event_anno) > 6:
                        event_prompt = "Events: ("
                        #l = [i for i in range(len(event_anno))]
                        #sample_idx = sample(l, 6)
                        l = [i for i in range(6)]
                        sample_idx = l
                        for j in sample_idx:
                            event_prompt += self.key2act[event_anno[j][0]]
                            event_prompt += ", "
                        event_prompt = event_prompt[:-2] + ")"
                        prompt += event_prompt
                    else:
                        event_prompt = "Events: ("
                        for j in event_anno:
                            event_prompt += self.key2act[j[0]]
                            event_prompt += ", "
                        event_prompt = event_prompt[:-2] + ")"
                        prompt += event_prompt
                    qa_prompt = prompt + ' Considering the information presented in the frame and list of events, select the correct answer from the options.'
                else:
                    qa_prompt = prompt + ' Considering the information presented in the frame, select the correct answer from the options.'
                #qa_prompt = prompt
                loc_prompt = 'Question: ' + q +  ' ' + hints + ' Does the information within the frame provide the necessary details to accurately answer the given question?'                
                answers = 'Option ' + ANS_MAPPING[int(ann['answer'])]
                duration = 1
            
            try:
                if 'VLEP' in qid:
                    video_id = ann['video']
                    if ':' in video_id:
                        # we set absolute path for vlep as it takes multiple video source
                        # you may change below paths to you own path
                        video_path = '/nas-hdd/shoubin/vlep_ytb_clips_tars/videos/vlep_ytb_clips/'
                    else:
                        video_id = video_id[:-3]
                        video_path = '/nas-hdd/shoubin/videos/tvqa/videos_3fps_with_audio/'
                    vpath = os.path.join(video_path, video_id + '.mp4')
                else:
                    vpath = os.path.join(self.vis_root, str(ann['video']) + '.mp4')   
                    
                frms, indices, fps = self.vis_processor(vpath, clip_proposal=clip) # vision feature extraction
                frms = frms.permute(1, 0, 2, 3)
                assert len(frms) == self.vis_processor.n_frms # 32 default
                
                if 'QVHighlight' in qid: 
                    time_stamp = [float(idx/fps) for idx in indices]
                    answers = []
                    if relevant_windows is not None:
                        for t in time_stamp:
                            flag = False
                            for span in relevant_windows:
                                if t >= float(span[0]) and t<= float(span[1]):
                                    answers.append('yes')
                                    flag = True 
                                    break
                            if not flag:
                                answers.append('no') 
                    else:
                        for t in time_stamp:
                            answers.append('no') # for test
                            
                    answers = '_'.join(answers)
                              
                result = True
            except Exception as e:
                
                print(f"Error while read file idx")
                print("video is: {}".format(ann['video']))
                index = random.randint(0, len(self.annotation) - 1)
            

            """
            ## event part added
            
            q_events = " | ".join(ann["q_events"])    
            
            ## load GT events
            event_anno = self.event_anno[qid]
            duration = event_anno['duration']
            events = event_anno['actions']
            span_list, hoi_list = [], []
            for e in events:
                span_list.append(e[-2:])
                hoi_list.append(e[0])
            if len(events) > self.max_windows:
                l = [i for i in range(len(events))]
                sample_idx = sample(l, self.max_windows)
                span_list = [span_list[idx] for idx in sample_idx]
                hoi_list = [hoi_list[idx] for idx in sample_idx]
            span = self.get_span_labels(span_list, duration, 10)    ## normalize times
            
            hoi_list = np.pad(hoi_list, (0, 10), 'constant', constant_values=-1)[:10]
        
            hoi = torch.Tensor(hoi_list).long()
            
            duration = 1
            """
            ### preprocess top-k on dataset (str 형태로 전달)
            topk = self.topk_events_anno[qid]

                
        return {
            "video": frms,
            "qa_input": qa_prompt,
            "loc_input": loc_prompt,
            "qa_output": answers,
            "question_id": "binary_"+qid,
            'duration': duration,
            #"q_events": q_events,
            #"span": span,
            #"hoi": hoi
            "topk_triplets": topk["t-k_trip"],
            "topk_span": np.array(topk["t-k_span"])
        }

    def collater(self, samples):
    #     q_events = []
    #     span = []
    #     hoi = []

    #     for i in range(len(samples)):
    #         span.append(samples[i].pop("span"))
    #         hoi.append(samples[i].pop("hoi"))

    #     samples = default_collate(samples)

    #     samples["q_events"] = q_events
    #     samples['span'] = span
    #     samples['hoi'] = hoi
        #print(samples)
        return default_collate(samples)
    