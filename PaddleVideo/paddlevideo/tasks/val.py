# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddlevideo.utils import get_logger
from ..loader.builder import build_dataloader, build_dataset
from ..metrics import build_metric
from ..modeling.builder import build_model
from paddlevideo.utils import load
import os, shutil

logger = get_logger("paddlevideo")


@paddle.no_grad()
def eval_one_model(model, data_loader, Metric):
    """Test model entry

    Args:
        cfg (dict): configuration.
        weights (str): weights path to load.
        parallel (bool): Whether to do multi-cards testing. Default: True.

    """

    model.eval()

    for batch_id, data in enumerate(data_loader):
        outputs = model(data, mode='test')
        Metric.update(batch_id, data, outputs)
    Metric.accumulate()
    if Metric.get_metrics:
        auc = Metric.auc
        ar_1 = Metric.ar_1
        ar_5 = Metric.ar_5
        ar_10 = Metric.ar_10
        ar_100 = Metric.ar_100

        return auc, ar_1, ar_5, ar_10, ar_100

@paddle.no_grad()
def eval_model(cfg, weights, parallel=True):
    # 1. Construct model.
    model = build_model(cfg.MODEL)
    if parallel:
        model = paddle.DataParallel(model)

    # 2. Construct dataset and dataloader.
    #cfg.DATASET.test.test_mode = True #2022/1/2 12:03
    #print(cfg.DATASET.test.test_mode)
    dataset = build_dataset((cfg.DATASET.test, cfg.PIPELINE.test))
    batch_size = cfg.DATASET.get("test_batch_size", 8)
    places = paddle.set_device('gpu')
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    dataloader_setting = dict(batch_size=batch_size,
                              num_workers=num_workers,
                              places=places,
                              drop_last=False,
                              shuffle=False)

    data_loader = build_dataloader(dataset, **dataloader_setting)
    # add params to metrics
    cfg.METRIC.data_size = len(dataset)
    cfg.METRIC.batch_size = batch_size
    Metric = build_metric(cfg.METRIC)
    model_paths = sorted([p for p in os.listdir(weights) if ".pdparams" in p])
    best_auc = 0
    metrics_results = []
    resume_epoch = cfg.get("resume_epoch", 0)
    for i, path in enumerate(model_paths):
        if i<resume_epoch:
            logger.info(
                f"| Skip {path}, continue... "
            )
            continue
        model_weight = os.path.join(weights, path)
        state_dicts = load(model_weight)
        model.set_state_dict(state_dicts)
        if os.path.exists(model_weight):
            print(f"Eval {path}...")
            auc, ar_1, ar_5, ar_10, ar_100 = eval_one_model(model, data_loader, Metric)
            if auc>best_auc:
                best_auc = auc
                best_ar = [ar_1, ar_5, ar_10, ar_100]
                best_model_path = path
            metrics_results.append((path, [auc, ar_1, ar_5, ar_10, ar_100]))
    print("The best model path:", best_model_path)
    print(f"AUC:{best_auc}, AR@1:{best_ar[0]}, AR@5:{best_ar[1]}, AR@10:{best_ar[2]}, AR@100:{best_ar[3]}")
    if resume_epoch==0:
        write_file = 'w'
    else:
        write_file = 'a'
    with open(os.path.join(weights, "eval_metrics_results.txt"), write_file) as f:
        f.write("AUC\tAR@1\tAR@5\tAR@10\tAR@100\n")
        for path, l in metrics_results:
            f.write(path+":\n")
            f.write(str(l[0])+'\t'+str(l[1])+'\t'+str(l[2])+'\t'+str(l[3])+'\t'+str(l[4])+'\n')
