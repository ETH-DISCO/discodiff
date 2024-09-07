import os
import json
import argparse

import numpy as np
from sklearn import metrics

def report_key_difference(gt_keys, eval_keys):
    if len(gt_keys) > 0 and len(eval_keys) > 0:
        print(">> With minor-major difference <<")
        confusion_matrix = metrics.confusion_matrix(gt_keys, eval_keys)
        print(confusion_matrix)
        classification_report = metrics.classification_report(gt_keys, eval_keys)
        print(classification_report)
        print(">> Without minor-major difference <<")
        gt_keys_merged = gt_keys % 12
        eval_keys_merged = eval_keys % 12
        confusion_matrix_reduced = metrics.confusion_matrix(gt_keys_merged, eval_keys_merged)
        print(confusion_matrix_reduced)
        classification_report_reduced= metrics.classification_report(gt_keys_merged, eval_keys_merged)
        print(classification_report_reduced)

        return confusion_matrix, confusion_matrix_reduced, classification_report, classification_report_reduced

def report_tempo_difference(gt_tempos, eval_tempos):
    if len(gt_tempos) > 0 and len(eval_tempos) > 0:
        print(">> With double-time difference <<")
        tempo_diff = (eval_tempos - gt_tempos)
        mean_tempo_diff = tempo_diff.mean()
        std_tempo_diff = tempo_diff.std()
        print("Tempo diff mean, std:", mean_tempo_diff, std_tempo_diff)
        print(">> Without double-time difference <<")
        eval_tempos_reduced = eval_tempos + 0
        gt_tempos_reduced = gt_tempos + 0
        eval_tempos_reduced[eval_tempos > 100] //= 2
        gt_tempos_reduced[gt_tempos > 100] //= 2
        tempo_diff = (eval_tempos_reduced - gt_tempos_reduced)
        mean_tempo_diff_reduced = tempo_diff.mean()
        std_tempo_diff_reduced = tempo_diff.std()
        print("Tempo diff mean, std:", mean_tempo_diff_reduced, std_tempo_diff_reduced)

        return mean_tempo_diff, mean_tempo_diff_reduced, std_tempo_diff, std_tempo_diff_reduced

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-gt-json-path", type=str)
    parser.add_argument("-eval-json-path", type=str)

    args = parser.parse_args()

    with open(args.eval_json_path, "r") as jsonFile:
        eval_dict = json.load(jsonFile)

    with open(args.gt_json_path, "r") as jsonFile:
        gt_dict = json.load(jsonFile)

    gt_keys_madmom = []
    gt_tempos_madmom = []
    gt_tempos_deeprhythm = []
    eval_keys_madmom = []
    eval_tempos_madmom = []
    eval_tempos_deeprhythm = []

    for file_id in eval_dict:
        if file_id in gt_dict:
            if "madmom_key" in gt_dict[file_id]:
                gt_keys_madmom.append(gt_dict[file_id]["madmom_key"])
            if "madmom_tempo" in gt_dict[file_id]:
                gt_tempos_madmom.append(gt_dict[file_id]["madmom_tempo"])
            if "deeprhythm_tempo" in gt_dict[file_id]:
                gt_tempos_deeprhythm.append(gt_dict[file_id]["deeprhythm_tempo"])
            if "madmom_key" in eval_dict[file_id]:
                eval_keys_madmom.append(eval_dict[file_id]["madmom_key"])
            if "madmom_tempo" in eval_dict[file_id]:
                eval_tempos_madmom.append(eval_dict[file_id]["madmom_tempo"])
            if "deeprhythm_tempo" in eval_dict[file_id]:
                eval_tempos_deeprhythm.append(eval_dict[file_id]["deeprhythm_tempo"])

    gt_keys_madmom = np.array(gt_keys_madmom)
    gt_tempos_madmom = np.array(gt_tempos_madmom)
    gt_tempos_deeprhythm = np.array(gt_tempos_deeprhythm)

    eval_keys_madmom = np.array(eval_keys_madmom)
    eval_tempos_madmom = np.array(eval_tempos_madmom)
    eval_tempos_deeprhythm = np.array(eval_tempos_deeprhythm)

    print("--- Key pred performance ---")
    report_key_difference(gt_keys_madmom, eval_keys_madmom)

    print("--- Tempo pred performance ---")
    report_tempo_difference(gt_tempos_madmom, eval_tempos_madmom)
    report_tempo_difference(gt_tempos_deeprhythm, eval_tempos_deeprhythm)
