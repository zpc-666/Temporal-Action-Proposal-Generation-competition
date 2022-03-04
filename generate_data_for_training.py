import json, copy
import argparse
from utils import win_splitting_per_video_for_training, win_splitting_per_video_for_testing

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parser_args():
    parser = argparse.ArgumentParser(description='Weight Decay Experiments')
    parser.add_argument('--video_info_path',
                        dest='video_info_path',
                        help='video information path',
                        default='/home/aistudio/data/data122998/label_cls14_train.json',
                        type=str)
    parser.add_argument('--feat_root',
                        dest='feat_root',
                        help='loading original video features dir',
                        default='/home/aistudio/data/data122998/Features_competition_train',
                        type=str)
    parser.add_argument('--train_feat_path',
                        dest='train_feat_path',
                        help='划分训练数据集视频之后切片的保存路径（每个切片包含至少一个完整的提案）',
                        default="/home/aistudio/data/pre_feat",
                        type=str)
    parser.add_argument('--train_json_path',
                        dest='train_json_path',
                        help='存放划分训练数据集视频之后的切片（每个切片包含至少一个完整的提案）相关的提案等信息的文件路径',
                        default='/home/aistudio/PaddleVideo/data/bmn/annotations_win.json',
                        type=str)
    parser.add_argument('--win_T',
                        dest='win_T',
                        help='划分训练数据集每个视频的窗口大小，步长默认为win_T//2',
                        default=100,
                        type=int)
    parser.add_argument('--val_gt_json_path',
                        dest='val_gt_json_path',
                        help='仅包含用于验证的视频切片（每个切片包含至少一个完整的提案）对应的提案等信息的文件路径',
                        default='/home/aistudio/PaddleVideo/data/bmn/annotations_val_gt.json',
                        type=str)
    parser.add_argument('--val_ratio',
                        dest='val_ratio',
                        help='训练集原未划分视频中val_ratio比例的视频切分后用于验证',
                        default=0.025,
                        type=float)
    parser.add_argument('--orig_val_gt_json_path',
                        dest='orig_val_gt_json_path',
                        help='仅包含用于验证的原未切分视频对应的提案等信息的文件路径',
                        default='/home/aistudio/PaddleVideo/data/bmn/annotations_val_gt_merged.json',
                        type=str)

    parser.add_argument('--val_feat_path',
                        dest='val_feat_path',
                        help='划分用于验证的视频之后切片的保存路径（每个切片不必包含提案），为了与测试集预测前数据划分一致',
                        default="/home/aistudio/data/val_feat",
                        type=str)
    parser.add_argument('--val_json_path',
                        dest='val_json_path',
                        help='包含划分用于验证的视频之后切片（每个切片不必包含提案）相关的信息的文件路径',
                        default='/home/aistudio/PaddleVideo/data/bmn/annotations_val_win.json',
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parser_args()
    print(args)

    # 将类文件对象中的JSON字符串直接转换成 Python 字典
    with open(args.video_info_path, 'r', encoding='utf-8') as f:
        label_dict = json.load(f)
        video_info = label_dict['gts']
        fps = label_dict['fps']
    
    print("准备划分并筛选得到用于训练验证的视频切片数据...")
    win_splitting_per_video_for_training(video_info=video_info,
                                        root=args.feat_root, 
                                        feat_output_path=args.train_feat_path, 
                                        train_json_path=args.train_json_path, 
                                        val_gt_json_path=args.val_gt_json_path,
                                        orig_val_gt_json_path=args.orig_val_gt_json_path,
                                        val_ratio=args.val_ratio,
                                        T=args.win_T, 
                                        fps=fps)
    

    with open(args.orig_val_gt_json_path, 'r', encoding='utf-8') as f:
        val_orig_dict = json.load(f)
    val_file_name = list(val_orig_dict['database'].keys())

    print("准备划分得到训练完用于验证模型AUC的视频切片数据...")
    win_splitting_per_video_for_testing(root=args.feat_root, 
                            feat_output_path=args.val_feat_path, 
                            json_path=args.val_json_path, 
                            test_file_name=val_file_name, 
                            T=args.win_T, 
                            fps=fps)


if __name__ == '__main__':
    main()