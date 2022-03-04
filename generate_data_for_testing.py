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
    parser.add_argument('--feat_root',
                        dest='feat_root',
                        help='loading original video features dir',
                        default='/home/aistudio/data/data123009/Features_competition_test_B',
                        type=str)
    parser.add_argument('--win_T',
                        dest='win_T',
                        help='划分训练数据集每个视频的窗口大小，步长默认为win_T//2',
                        default=100,
                        type=int)

    parser.add_argument('--fps',
                        dest='fps',
                        help='视频的fps',
                        default=25,
                        type=int)

    parser.add_argument('--test_feat_path',
                        dest='test_feat_path',
                        help='划分用于测试的视频之后切片的保存路径（每个切片不必包含提案）',
                        default="/home/aistudio/data/test_feat",
                        type=str)
    parser.add_argument('--test_json_path',
                        dest='test_json_path',
                        help='包含用于测试的视频划分之后切片（每个切片不必包含提案）相关信息的文件路径',
                        default='/home/aistudio/PaddleVideo/data/bmn/annotations_test.json',
                        type=str)

    args = parser.parse_args()

    return args


def main():
    args = parser_args()
    print(args)

    print("准备划分得到测试集视频的切片数据...")
    win_splitting_per_video_for_testing(root=args.feat_root, 
                            feat_output_path=args.test_feat_path, 
                            json_path=args.test_json_path, 
                            test_file_name=None, 
                            T=args.win_T, 
                            fps=args.fps)


if __name__ == '__main__':
    main()