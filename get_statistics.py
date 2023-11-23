import pandas as pd
import os
import pickle
import argparse
import numpy as np
import json


def compare_classification_score(score1, score2):
    return score1 > score2


def compare_regression_score(score1, score2):
    return score1 < score2


def get_scores(checkpoint_paths):
    best_scores = {}
    all_scores = {}
    for checkpoint_path in checkpoint_paths:
        i_chpt = int(checkpoint_path.split('/')[-1].split('_')[-1])
        scores = {}
        has_score = False
        for file in os.listdir(f"{checkpoint_path}"):
            if 'eval_res' in file:
                has_score = True
                with open(f"{checkpoint_path}/{file}") as f:
                    lines = f.readlines()
                    for line in lines:
                        if 'Scores:' in line:
                            splits = line.split(' Scores: ')
                            score = float(splits[-1])
                            scores[splits[0]] = score
        if has_score:
            all_scores[i_chpt] = scores
            if len(best_scores) == 0 or score_comparer(scores['Mean'], best_scores['Mean']):
                best_scores['Mean'] = scores['Mean']
                best_scores['Std'] = scores['Std']
                best_scores['i'] = i_chpt
    if len(best_scores) == 0:
        best_scores = {'Mean': None, 'Std': None, 'i': None}
    return best_scores, all_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Statistics")

    parser.add_argument('--folder', type=str, default='./checkpoint')
    parser.add_argument('--single', action="store_true")
    args = parser.parse_args()

    current_folder = os.getcwd()

    dataset_name, model_type, folder_name = args.folder.split('/')[-3:]
    from utils.dataset import GRAPH_CLASSIFICATION_DATASETS, GRAPH_REGRESSION_DATASETS, SIMPLE_REGRESSION_DATASETS

    score_comparer = compare_regression_score if dataset_name in GRAPH_REGRESSION_DATASETS or \
                                                 dataset_name in SIMPLE_REGRESSION_DATASETS \
        else compare_classification_score
    print(score_comparer)
    folder_paths = []
    if not args.single:
        for file in os.listdir(f"{args.folder}"):
            if os.path.isdir(f"{args.folder}/{file}") and "train_opti" in file:
                checkpoint_paths = []
                for file1 in os.listdir(f"{args.folder}/{file}"):
                    if os.path.isdir(f"{args.folder}/{file}/{file1}") and 'checkpoint' in file1:
                        checkpoint_paths.append(f"{args.folder}/{file}/{file1}")

                if len(checkpoint_paths) > 0:
                    folder_paths.append((f"{args.folder}/{file}", checkpoint_paths))
        names = []
        all_results = []
        for folder_path, checkpoint_paths in folder_paths:
            model_name = folder_path.split('/')[-1]
            params_path = folder_path + '/params.json'
            with open(params_path) as json_file:
                data = json.load(json_file)
            add_feature = data['add_feature']
            batch_size = data['batch_size']
            beta = data['beta']
            lr = data['lr']
            noise = data['noise']
            noise_x = data['noise_x']
            split_graph_dim = data['split_graph_dim']
            var = data['var']

            best_scores, all_scores = get_scores(checkpoint_paths)

            if len(names) == 0:
                names = ['Name', 'Batch size', 'LR', 'Noise', 'Noise_x', 'Split_Graph_Dim', 'Beta', 'Var',
                         'Add_Feature',
                         *best_scores.keys()]
            all_results.append(
                [model_name, batch_size, lr, noise, noise_x, split_graph_dim, beta, var, add_feature,
                 *best_scores.values()])
        df = pd.DataFrame(np.array(all_results), columns=names)
        df.to_csv(f'{args.folder}/stats.csv', index=False)
    else:
        checkpoint_paths = []
        for file1 in os.listdir(f"{args.folder}"):
            if os.path.isdir(f"{args.folder}/{file1}") and 'checkpoint' in file1:
                checkpoint_paths.append(f"{args.folder}/{file1}")

        if len(checkpoint_paths) > 0:
            folder_paths.append((f"{args.folder}", checkpoint_paths))

        best_scores, all_scores = get_scores(checkpoint_paths)
        sorted_scores = dict(sorted(all_scores.items()))
        print(f'Best Score : {best_scores} \n'
              f'All Score : {sorted_scores}')
