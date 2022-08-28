import argparse
import logging
import os
import sys
from os.path import join
from pathlib import Path

import pandas as pd

import compute_metrics

sys.path.append(r"/media/medical/gasperp/projects")
from utilities import utilities


def main():
    # Set parser
    parser = argparse.ArgumentParser(
        prog="filter LR",
        description="tool for postprocessing left and right laterality organs",
    )
    parser.add_argument(
        "-gt",
        "--gt_dir",
        type=str,
        required=True,
        help="absolute path to input segmentations directory",
    )
    parser.add_argument(
        "-pred",
        "--pred_dir",
        type=str,
        required=True,
        help="absolute path to input segmentations directory",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default=None,
        help="absolute path to results csv",
    )
    parser.add_argument(
        "--gt_dataset_json",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--pred_dataset_json",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--model_task_number",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--model_task_name",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--dataset_task_number",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--dataset_task_name",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--trainer_class",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--plans_name",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--prediction_mode",
        type=str,
        default=None,
        help="absolute path to dataset json",
    )
    parser.add_argument(
        "--get_settings_from_dir_name",
        default=False,
        action="store_true",
        help="Parses folder name to get settings",
    )

    # running in terminal
    args = vars(parser.parse_args())

    if args["results_csv"] is None:
        args["results_csv"] = join(args["pred_dir"], "results.csv")

    gt_seg_fps = utilities.list_all_files_in_dir_and_subdir(
        args["gt_dir"], suffix=".nii.gz", exclude="imagesT", sort_by_filename=True
    )
    pred_seg_fps = utilities.list_all_files_in_dir_and_subdir(
        args["pred_dir"], suffix=".nii.gz", sort_by_filename=True
    )

    gt_dataset_dict = utilities.read_dict_in_json(args["gt_dataset_json"])
    gt_lbl_dict = {
        o_name: int(lbl) for lbl, o_name in gt_dataset_dict["labels"].items()
    }
    pred_dataset_dict = utilities.read_dict_in_json(args["pred_dataset_json"])
    pred_lbl_dict = {
        o_name: int(lbl) for lbl, o_name in pred_dataset_dict["labels"].items()
    }

    if args["get_settings_from_dir_name"]:
        settings_str = Path(args["pred_dir"]).name
        args["fold"] = settings_str.split("FOLD-")[1].split("_TRAINER")[0]
        args["trainer_class"] = settings_str.split("TRAINER-")[1].split("_PLANS")[0]
        args["plans_name"] = settings_str.split("PLANS-")[1].split("_CHK")[0]
        args["checkpoint"] = settings_str.split("CHK-")[1].split("_DATASET")[0]
        args["dataset_task_number"] = settings_str.split("DATASET-")[1].split("_")[0]
        args["dataset_task_name"] = Path(args["gt_dir"]).name
        args["model_task_name"] = (
            args["pred_dir"]
            .replace("/media/medical/projects/head_and_neck/nnUnet/", "")
            .split("/")[0]
        )
        args["model_task_number"] = (
            args["model_task_name"].replace("Task", "").split("_")[0]
        )

    settings_info = {
        "model_task_number": args.get("model_task_number"),
        "model_task_name": args.get("model_task_name"),
        "dataset_task_number": args.get("dataset_task_number"),
        "dataset_task_name": args.get("dataset_task_name"),
        "fold": args.get("fold"),
        "trainer_class": args.get("trainer_class"),
        "plans_name": args.get("plans_name"),
        "checkpoint": args.get("checkpoint"),
        "prediction_mode": args.get("prediction_mode"),
    }
    dfs = []

    compute = compute_metrics.compute_metrics_deepmind(
        organs_labels_dict_gt=gt_lbl_dict,
        organs_labels_dict_pred=pred_lbl_dict,
    )

    for gt_fp, pred_fp in zip(gt_seg_fps, pred_seg_fps):
        assert Path(gt_fp).name == Path(pred_fp).name, "wrong pair"

        out_dict_tmp = compute.execute(fpath_gt=gt_fp, fpath_pred=pred_fp)

        df = pd.DataFrame.from_dict(out_dict_tmp)
        for k, val in settings_info.items():
            df[k] = val
        dfs.append(df)

    csv_path = args["results_csv"]
    if os.path.exists(csv_path):
        logging.info(
            f"Found existing .csv file on location {csv_path}, merging existing and new dataframe"
        )
        existing_df = [pd.read_csv(csv_path, index_col=0)] + dfs
        pd.concat(existing_df, ignore_index=True).to_csv(csv_path)
    else:
        pd.concat(dfs, ignore_index=True).to_csv(csv_path)


if __name__ == "__main__":
    main()
