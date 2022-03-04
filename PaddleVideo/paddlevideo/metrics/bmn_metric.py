# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import os, pickle, tqdm
import json
import numpy as np
import pandas as pd
import multiprocessing as mp

from .registry import METRIC
from .base import BaseMetric
from .ActivityNet import ANETproposal
from paddlevideo.utils import get_logger

logger = get_logger("paddlevideo")


def iou_with_anchors(anchors_min, anchors_max, box_min, box_max):
    """Compute jaccard score between a box and the anchors.
    """
    len_anchors = anchors_max - anchors_min
    int_xmin = np.maximum(anchors_min, box_min)
    int_xmax = np.minimum(anchors_max, box_max)
    inter_len = np.maximum(int_xmax - int_xmin, 0.)
    union_len = len_anchors - inter_len + box_max - box_min
    jaccard = np.divide(inter_len, union_len)
    return jaccard


def boundary_choose(score_list):
    """Choose start and end boundary from score.
    """
    max_score = max(score_list)
    mask_high = (score_list > max_score * 0.5)
    score_list = list(score_list)
    score_middle = np.array([0.0] + score_list + [0.0])
    score_front = np.array([0.0, 0.0] + score_list)
    score_back = np.array(score_list + [0.0, 0.0])
    mask_peak = ((score_middle > score_front) & (score_middle > score_back))
    mask_peak = mask_peak[1:-1]
    mask = (mask_high | mask_peak).astype('float32')
    return mask


def soft_nms(df, alpha, t1, t2):
    '''
    df: proposals generated by network;
    alpha: alpha value of Gaussian decaying function;
    t1, t2: threshold for soft nms.
    '''
    df = df.sort_values(by="score", ascending=False)
    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 101:
        max_index = tscore.index(max(tscore))
        tmp_iou_list = iou_with_anchors(np.array(tstart), np.array(tend),
                                        tstart[max_index], tend[max_index])
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = tend[max_index] - tstart[max_index]
                if tmp_iou > t1 + (t2 - t1) * tmp_width:
                    tscore[idx] = tscore[idx] * np.exp(
                        -np.square(tmp_iou) / alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf

def soft_nms_for_merging(proposal_dict, alpha=0.4, t1=0.55, t2=0.9, dscale=4):
    '''
    proposal_dict: proposals generated by network;
    alpha: alpha value of Gaussian decaying function;
    t1, t2: threshold for soft nms.
    '''
    #df = df.sort_values(by="score", ascending=False)
    sorted_proposal = sorted(proposal_dict, key=lambda x:x["score"], reverse=True)
    tstart = []
    tend = []
    tscore = []
    for pp in sorted_proposal:
        tstart.append(pp["segment"][0])
        tend.append(pp["segment"][1])
        tscore.append(pp["score"])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < 101:
        max_index = tscore.index(max(tscore))
        tmp_iou_list = iou_with_anchors(np.array(tstart), np.array(tend),
                                        tstart[max_index], tend[max_index])
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = tmp_iou_list[idx]
                tmp_width = (tend[max_index] - tstart[max_index])/dscale
                if tmp_iou > t1 + (t2 - t1) * tmp_width:
                    tscore[idx] = tscore[idx] * np.exp(
                        -np.square(tmp_iou) / alpha)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    new_proposal = []
    for i in range(len(rscore)):
        pp = {}
        pp['score'] = round(rscore[i], 2)
        pp["segment"] = [round(rstart[i], 2), round(rend[i], 2)]
        new_proposal.append(pp)
    
    return new_proposal

@METRIC.register
class BMNMetric(BaseMetric):
    """
    Metrics for BMN. Two Stages in this metric:
    (1) Get test results using trained model, results will be saved in BMNMetric.result_path;
    (2) Calculate metrics using results file from stage (1).
    """
    def __init__(self,
                 data_size,
                 batch_size,
                 tscale,
                 dscale,
                 file_path,
                 ground_truth_filename,
                 subset,
                 output_path,
                 result_path,
                 get_metrics=True,
                 log_interval=100,
                 to_merge=False):
        """
        Init for BMN metrics.
        Params:
            get_metrics: whether to calculate AR@N and AUC metrics or not, default True.
        """
        super().__init__(data_size, batch_size, log_interval)
        assert self.batch_size == 1, " Now we just support batch_size==1 test"
        assert self.world_size == 1, " Now we just support single-card test"

        self.tscale = tscale
        self.dscale = dscale
        self.file_path = file_path
        self.ground_truth_filename = ground_truth_filename
        self.subset = subset
        self.output_path = output_path
        self.result_path = result_path
        self.get_metrics = get_metrics
        self.to_merge = to_merge

        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)
        if not os.path.isdir(self.result_path):
            os.makedirs(self.result_path)

        self.video_dict, self.video_list = self.get_dataset_dict(
            self.file_path, self.subset)

    def get_dataset_dict(self, file_path, subset):
        annos = json.load(open(file_path))
        video_dict = {}
        for video_name in annos.keys():
            video_subset = annos[video_name]["subset"]
            if subset in video_subset:
                video_dict[video_name] = annos[video_name]
        video_list = list(video_dict.keys())
        video_list.sort()
        return video_dict, video_list

    def update(self, batch_id, data, outputs):
        """update metrics during each iter
        """
        fid = data[4].numpy()
        pred_bm, pred_start, pred_end = outputs
        pred_bm = pred_bm.numpy()
        pred_start = pred_start[0].numpy()
        pred_end = pred_end[0].numpy()

        snippet_xmins = [1.0 / self.tscale * i for i in range(self.tscale)]
        snippet_xmaxs = [
            1.0 / self.tscale * i for i in range(1, self.tscale + 1)
        ]
        cols = ["xmin", "xmax", "score"]

        video_name = self.video_list[fid[0]]
        pred_bm = pred_bm[0, 0, :, :] * pred_bm[0, 1, :, :]
        start_mask = boundary_choose(pred_start)
        start_mask[0] = 1.
        end_mask = boundary_choose(pred_end)
        end_mask[-1] = 1.
        score_vector_list = []
        for idx in range(self.dscale):
            for jdx in range(self.tscale):
                start_index = jdx
                end_index = start_index + idx
                if end_index < self.tscale and start_mask[
                        start_index] == 1 and end_mask[end_index] == 1:
                    xmin = snippet_xmins[start_index]
                    xmax = snippet_xmaxs[end_index]
                    xmin_score = pred_start[start_index]
                    xmax_score = pred_end[end_index]
                    bm_score = pred_bm[idx, jdx]
                    conf_score = xmin_score * xmax_score * bm_score
                    score_vector_list.append([xmin, xmax, conf_score])

        score_vector_list = np.stack(score_vector_list)
        video_df = pd.DataFrame(score_vector_list, columns=cols)
        video_df.to_csv(os.path.join(self.output_path, "%s.csv" % video_name),
                        index=False)

        if batch_id % self.log_interval == 0:
            logger.info("Processing................ batch {}".format(batch_id))

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        # check clip index of each video
        #Stage1
        self.bmn_post_processing(self.video_dict, self.subset, self.output_path,
                                 self.result_path)
        if self.get_metrics:
            result_path = os.path.join(self.result_path, "bmn_results_validation.json")
            if self.to_merge:
                merged_result_path = os.path.join(self.result_path, "bmn_merged_results_validation.json")
                self.merging_output_per_video(self.tscale, self.ground_truth_filename, 
                                        result_path,                  
                                        merged_result_path)
                result_path = merged_result_path

            logger.info("[TEST] calculate metrics...")
            #Stage2
            uniform_average_nr_proposals_valid, uniform_average_recall_valid, uniform_recall_valid = self.cal_metrics(
                self.ground_truth_filename,
                result_path,
                max_avg_nr_proposals=100,
                tiou_thresholds=np.linspace(0.5, 0.9, 9), # orig: (0.5, 0.95, 10)
                subset='validation')
            logger.info("AR@1; AR@5; AR@10; AR@100")
            self.ar_1 = 100 * np.mean(uniform_recall_valid[:, 0])
            self.ar_5 = 100 * np.mean(uniform_recall_valid[:, 4])
            self.ar_10 = 100 * np.mean(uniform_recall_valid[:, 9])
            self.ar_100 = 100 * np.mean(uniform_recall_valid[:, -1])
            logger.info("%.02f %.02f %.02f %.02f" %
                        (self.ar_1,
                         self.ar_5,
                         self.ar_10,
                         self.ar_100))
            self.auc = int(np.trapz(uniform_average_recall_valid, uniform_average_nr_proposals_valid)*100)/100.

    def bmn_post_processing(self, video_dict, subset, output_path, result_path):
        video_list = list(video_dict.keys())
        global result_dict
        result_dict = mp.Manager().dict()
        pp_num = 12

        num_videos = len(video_list)
        num_videos_per_thread = int(num_videos / pp_num)
        processes = []
        for tid in range(pp_num - 1):
            tmp_video_list = video_list[tid * num_videos_per_thread:(tid + 1) *
                                        num_videos_per_thread]
            p = mp.Process(target=self.video_process,
                           args=(tmp_video_list, video_dict, output_path,
                                 result_dict))
            p.start()
            processes.append(p)
        tmp_video_list = video_list[(pp_num - 1) * num_videos_per_thread:]
        p = mp.Process(target=self.video_process,
                       args=(tmp_video_list, video_dict, output_path,
                             result_dict))
        p.start()
        processes.append(p)
        for p in processes:
            p.join()

        result_dict = dict(result_dict)
        output_dict = {
            "version": "VERSION 1.3",
            "results": result_dict,
            "external_data": {}
        }
        outfile = open(
            os.path.join(result_path, "bmn_results_%s.json" % subset), "w")

        json.dump(output_dict, outfile)
        outfile.close()

    def video_process(self,
                      video_list,
                      video_dict,
                      output_path,
                      result_dict,
                      snms_alpha=0.4,
                      snms_t1=0.55,
                      snms_t2=0.9):

        for vidx, video_name in enumerate(video_list):
            if vidx % self.log_interval == 0:
                logger.info("Processing video........" + video_name)
            df = pd.read_csv(os.path.join(output_path, video_name + ".csv"))
            if len(df) > 1:
                df = soft_nms(df, snms_alpha, snms_t1, snms_t2)

            video_duration = video_dict[video_name]["duration_second"]
            proposal_list = []
            for idx in range(min(100, len(df))):
                tmp_prop={"score":df.score.values[idx], \
                          "segment":[max(0,df.xmin.values[idx])*video_duration, \
                                     min(1,df.xmax.values[idx])*video_duration]}
                proposal_list.append(tmp_prop)
            result_dict[video_name[2:]] = proposal_list

    def cal_metrics(self,
                    ground_truth_filename,
                    proposal_filename,
                    max_avg_nr_proposals=100,
                    tiou_thresholds=np.linspace(0.5, 0.95, 10),
                    subset='validation'):

        anet_proposal = ANETproposal(ground_truth_filename,
                                     proposal_filename,
                                     tiou_thresholds=tiou_thresholds,
                                     max_avg_nr_proposals=max_avg_nr_proposals,
                                     subset=subset,
                                     verbose=True,
                                     check_status=False)
        anet_proposal.evaluate()
        recall = anet_proposal.recall
        average_recall = anet_proposal.avg_recall
        average_nr_proposals = anet_proposal.proposals_per_video

        return (average_nr_proposals, average_recall, recall)

    def merging_output_per_video(self, 
                                win_t, 
                                ground_truth_filename, 
                                proposal_filename, 
                                merging_output_filename,
                                snms_alpha=0.4, 
                                snms_t1=0.55,
                                snms_t2=0.9):
        # 合并回提交文件
        with open(ground_truth_filename, 'r', encoding='utf-8') as f:
            label_dict = json.load(f)
        val_file_name = label_dict['database'].keys()
        fps = label_dict['fps']

        with open(proposal_filename, 'r', encoding='utf-8') as f:
            pred_dict = json.load(f)

        new_pred_dict = {"version":pred_dict["version"], "external_data": {}}
        results_dict = pred_dict["results"]
        new_results_dict = {}
        for file_name in tqdm.tqdm(val_file_name):
            frames_len = label_dict['database'][file_name]['num_frames']
            clip_count = 1
            proposal = []
            for start_f in range(0, frames_len, win_t//2):
                end_f = start_f+win_t
                if end_f<frames_len:
                    start_second = start_f/fps

                else:
                    start_second = (frames_len-win_t)/fps

                clip_name = file_name[:-4]+"_clip"+str(clip_count)+".mp4" 
                if clip_name[2:] not in results_dict.keys(): #bmn_metric.py 第279行
                    clip_count += 1
                    continue
                if len(results_dict[clip_name[2:]])==0:
                    clip_count += 1
                    continue   
                for pp in results_dict[clip_name[2:]]:
                    pp["segment"][0] = pp["segment"][0]+start_second
                    pp["segment"][1] = pp["segment"][1]+start_second
                    proposal.append(pp)
                clip_count += 1
    
            proposal = soft_nms_for_merging(proposal, snms_alpha, snms_t1, snms_t2)
            proposal = sorted(proposal, key=lambda x:x["segment"][0], reverse=False)
            new_results_dict[file_name[2:]] = proposal

        new_pred_dict["results"] = new_results_dict
        with open(merging_output_filename, 'w') as f:
            json.dump(new_pred_dict, f)

