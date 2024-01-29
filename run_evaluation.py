import itertools
import argparse
import datetime
import os

import glob
import numpy as np

import shutil
import csv
from vivid123.metrics import LPIPSMeter, PSNRMeter, SSIM, FOR

SLURM_TMPDIR = (
    os.getenv("SLURM_TMPDIR")
    if os.getenv("SLURM_TMPDIR")
    else "/scratch/rendering-360/"  # the dir where the gt images are decompressed to, if it exists on your local machine
)

# should specify the indeces of the frames to be evaluated in both the generation dir and the gt dir, like the example below
eval_specs = [
    {
        "exp_name": "num_frames_24",
        "vid_frame_indeces": [2 * i for i in range(12)],
        "gt_indeces": [3 + 6 * i for i in range(12)],
    },
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # For batch running on Compute Canada
    parser.add_argument("--exp_id", type=int)
    parser.add_argument(
        "--metadata",
        type=str,
        default="scripts/gso_metadata_object_prompt_100.csv",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default="/scratch/rendering-360-zips",
        help="Directory containing the ground truth images, each object in a separate zip file, the zip file contains a folder named 'img' with the images.",
    )

    args = parser.parse_args()

    lpips_scorer = LPIPSMeter(device="cuda:0", size=512, net="vgg")
    psnr_scorer = PSNRMeter(size=512)
    ssim_scorer = SSIM(size=512)
    for_scorer = FOR(
        size=512,
    )

    exp = eval_specs[args.exp_id]["exp_name"]
    expdir = f"/scratch-ssd/vivid123/exps/samples/{exp}"
    savedir = f"/scratch-ssd/vivid123/exps/evaluations/{exp}"
    os.makedirs(savedir, exist_ok=True)

    vid_frame_indeces = eval_specs[args.exp_id]["vid_frame_indeces"]
    gt_indeces = eval_specs[args.exp_id]["gt_indeces"]
    num_views = len(vid_frame_indeces)

    csv_columns = (
        ["obj", "psnr", "lpips", "ssim", "for_8", "for_16"]
        + [f"psnr_{i}" for i in range(num_views)]
        + [f"lpips_{i}" for i in range(num_views)]
        + [f"ssim_{i}" for i in range(num_views)]
        + [f"for_8_{i}" for i in range(num_views)]
        + [f"for_16_{i}" for i in range(num_views)]
    )

    with open(args.metadata, newline="") as csvmetadatafile:
        csv_lines = csv.reader(csvmetadatafile, delimiter=",", quotechar='"')
        for csv_line in csv_lines:
            object_identifier = csv_line[0]
            csv_exp_file = f"{savedir}/{object_identifier}.csv"
            if os.path.isfile(csv_exp_file):
                continue

            if not os.path.isfile(f"{SLURM_TMPDIR}/{object_identifier}/img/000.png"):
                shutil.unpack_archive(
                    f"{args.gt_dir}/{object_identifier}.zip",
                    f"{SLURM_TMPDIR}/{object_identifier}",
                )

            result_dict = {}

            gt_paths_sorted = sorted(
                glob.glob(f"{SLURM_TMPDIR}/{object_identifier}/img/*.png")
            )
            pred_paths_sorted = sorted(
                glob.glob(f"{expdir}/{object_identifier}/xl_frames/*.png")
            )
            print(f"object_identifier: {object_identifier}")
            gt_paths = [gt_paths_sorted[i] for i in gt_indeces]
            pred_paths = [pred_paths_sorted[i] for i in vid_frame_indeces]
            if (
                len(gt_paths) == 0
                or len(pred_paths) == 0
                or len(gt_paths) != len(pred_paths)
            ):
                print(f"gt_path_list: {gt_paths}")
                print(f"pred_path_list: {pred_paths}")
                print(
                    f"\n\n{object_identifier} doesn't have data or the rendering wasn't complete in {expdir}! Skipping this object...\n\n"
                )
                continue

            result_dict["obj"] = object_identifier
            result_dict["psnr"], psnrs = psnr_scorer.score_gt(gt_paths, pred_paths)
            result_dict["lpips"], lpips = lpips_scorer.score_gt(gt_paths, pred_paths)
            result_dict["ssim"], ssims = ssim_scorer.score_gt(gt_paths, pred_paths)
            masked_flow_error_and_mask = for_scorer.raft_predict(
                gt_paths,
                pred_paths,
                results_path=f"exps/optical_flow_tmp/{exp}",
                obj_id=object_identifier,
            )
            result_dict["for_8"], for_8s = for_scorer.score_gt(
                masked_flow_error_and_mask, threshold=8
            )
            result_dict["for_16"], for_16s = for_scorer.score_gt(
                masked_flow_error_and_mask, threshold=16
            )

            for i in range(num_views):
                result_dict[f"psnr_{i}"] = psnrs[i]
                result_dict[f"lpips_{i}"] = lpips[i]
                result_dict[f"ssim_{i}"] = ssims[i]
                result_dict[f"for_8_{i}"] = for_8s[i]
                result_dict[f"for_16_{i}"] = for_16s[i]

            print(f"PSNR for {object_identifier}: {result_dict['psnr']}")
            print(f"LPIPS for {object_identifier}: {result_dict['lpips']}")
            print(f"SSIM for {object_identifier}: {result_dict['ssim']}")
            print(f"FOR_8 for {object_identifier}: {result_dict['for_8']}")
            print(f"FOR_16 for {object_identifier}: {result_dict['for_16']}")

            with open(csv_exp_file, "a") as csvexpfile:
                writer = csv.DictWriter(csvexpfile, fieldnames=csv_columns)
                writer.writeheader()
                writer.writerow(result_dict)
