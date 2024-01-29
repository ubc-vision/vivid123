import csv
import itertools
import argparse
import numpy as np
import os

eval_specs = [
    {"exp_name": "num_frames_12", "num_frames": 12},
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=str,
        default="/scratch-ssd/vivid123/scripts/gso_metadata_object_prompt_100.csv",
    )

    args = parser.parse_args()

    psnr_save_dir = f"exps/stats/total_result_metrics_psnr"
    lpips_save_dir = f"exps/stats/total_result_metrics_lpips"
    ssim_save_dir = f"exps/stats/total_result_metrics_ssim"
    for_8_save_dir = f"exps/stats/total_result_metrics_for_8"
    for_16_save_dir = f"exps/stats/total_result_metrics_for_16"
    os.makedirs(psnr_save_dir, exist_ok=True)
    os.makedirs(lpips_save_dir, exist_ok=True)
    os.makedirs(ssim_save_dir, exist_ok=True)
    os.makedirs(for_8_save_dir, exist_ok=True)
    os.makedirs(for_16_save_dir, exist_ok=True)

    for eval_spec in eval_specs:
        exp_name = eval_spec["exp_name"]
        num_views = eval_spec["num_frames"]
        eval_dir = f"exps/evaluations/{exp_name}"

        # Aggregate the evaluation results to the final stats
        csv_total_file_psnr = f"{psnr_save_dir}/{exp_name}.csv"
        csv_total_columns_psnr = ["exp_name", "psnr"] + [
            f"psnr_{i}" for i in range(num_views)
        ]
        with open(csv_total_file_psnr, "w") as csvtotalfile:
            writer = csv.DictWriter(csvtotalfile, fieldnames=csv_total_columns_psnr)
            writer.writeheader()

        csv_total_file_lpips = f"{lpips_save_dir}/{exp_name}.csv"
        csv_total_columns_lpips = ["exp_name", "lpips"] + [
            f"lpips_{i}" for i in range(num_views)
        ]
        with open(csv_total_file_lpips, "w") as csvtotalfile:
            writer = csv.DictWriter(csvtotalfile, fieldnames=csv_total_columns_lpips)
            writer.writeheader()

        csv_total_file_ssim = f"{ssim_save_dir}/{exp_name}.csv"
        csv_total_columns_ssim = ["exp_name", "ssim"] + [
            f"ssim_{i}" for i in range(num_views)
        ]
        with open(csv_total_file_ssim, "w") as csvtotalfile:
            writer = csv.DictWriter(csvtotalfile, fieldnames=csv_total_columns_ssim)
            writer.writeheader()

        csv_total_file_for_8 = f"{for_8_save_dir}/{exp_name}.csv"
        csv_total_columns_for_8 = ["exp_name", "for_8"] + [
            f"for_8_{i}" for i in range(num_views)
        ]
        with open(csv_total_file_for_8, "w") as csvtotalfile:
            writer = csv.DictWriter(csvtotalfile, fieldnames=csv_total_columns_for_8)
            writer.writeheader()

        csv_total_file_for_16 = f"{for_16_save_dir}/{exp_name}.csv"
        csv_total_columns_for_16 = ["exp_name", "for_16"] + [
            f"for_16_{i}" for i in range(num_views)
        ]
        with open(csv_total_file_for_16, "w") as csvtotalfile:
            writer = csv.DictWriter(csvtotalfile, fieldnames=csv_total_columns_for_16)
            writer.writeheader()

        results_psnr = {"psnr": []}
        results_lpips = {"lpips": []}
        results_ssim = {"ssim": []}
        results_for_8 = {"for_8": []}
        results_for_16 = {"for_16": []}
        for i in range(num_views):
            results_psnr[f"psnr_{i}"] = []
            results_lpips[f"lpips_{i}"] = []
            results_ssim[f"ssim_{i}"] = []
            results_for_8[f"for_8_{i}"] = []
            results_for_16[f"for_16_{i}"] = []

        count = 0
        with open(args.metadata, newline="") as csvmetadatafile:
            csv_lines = csv.reader(csvmetadatafile, delimiter=",", quotechar='"')
            for csv_line in csv_lines:
                object_identifier = csv_line[0]
                if not os.path.isfile(f"{eval_dir}/{object_identifier}.csv"):
                    print(
                        f"WARNING: {exp_name} doesn't have {object_identifier}! Skipping this object..."
                    )
                    continue

                count += 1
                with open(
                    f"{eval_dir}/{object_identifier}.csv", newline=""
                ) as csv_object_metric_file:
                    reader = csv.DictReader(csv_object_metric_file, delimiter=",")
                    row = reader.__next__()
                    print(row)
                    results_psnr["psnr"].append(float(row["psnr"]))
                    results_lpips["lpips"].append(float(row["lpips"]))
                    results_ssim["ssim"].append(float(row["ssim"]))
                    results_for_8["for_8"].append(float(row["for_8"]))
                    results_for_16["for_16"].append(float(row["for_16"]))
                    for i in range(num_views):
                        results_psnr[f"psnr_{i}"].append(float(row[f"psnr_{i}"]))
                        results_lpips[f"lpips_{i}"].append(float(row[f"lpips_{i}"]))
                        results_ssim[f"ssim_{i}"].append(float(row[f"ssim_{i}"]))
                        results_for_8[f"for_8_{i}"].append(float(row[f"for_8_{i}"]))
                        results_for_16[f"for_16_{i}"].append(float(row[f"for_16_{i}"]))

            print(f"{exp_name} has {count} objects finished")

            # write into csv file
            with open(csv_total_file_psnr, "a") as csvtotalfile:
                writer = csv.DictWriter(csvtotalfile, fieldnames=csv_total_columns_psnr)
                row_to_write = {}
                row_to_write["exp_name"] = exp_name
                row_to_write["psnr"] = np.mean(results_psnr["psnr"])
                for i in range(num_views):
                    row_to_write[f"psnr_{i}"] = np.mean(results_psnr[f"psnr_{i}"])
                writer.writerow(row_to_write)

            with open(csv_total_file_lpips, "a") as csvtotalfile:
                writer = csv.DictWriter(
                    csvtotalfile, fieldnames=csv_total_columns_lpips
                )
                row_to_write = {}
                row_to_write["exp_name"] = exp_name
                row_to_write["lpips"] = np.mean(results_lpips["lpips"])
                for i in range(num_views):
                    row_to_write[f"lpips_{i}"] = np.mean(results_lpips[f"lpips_{i}"])
                writer.writerow(row_to_write)

            with open(csv_total_file_ssim, "a") as csvtotalfile:
                writer = csv.DictWriter(csvtotalfile, fieldnames=csv_total_columns_ssim)
                row_to_write = {}
                row_to_write["exp_name"] = exp_name
                row_to_write["ssim"] = np.mean(results_ssim["ssim"])
                for i in range(num_views):
                    row_to_write[f"ssim_{i}"] = np.mean(results_ssim[f"ssim_{i}"])
                writer.writerow(row_to_write)

            with open(csv_total_file_for_8, "a") as csvtotalfile:
                writer = csv.DictWriter(
                    csvtotalfile, fieldnames=csv_total_columns_for_8
                )
                row_to_write = {}
                row_to_write["exp_name"] = exp_name
                row_to_write["for_8"] = np.mean(results_for_8["for_8"])
                for i in range(num_views):
                    row_to_write[f"for_8_{i}"] = np.mean(results_for_8[f"for_8_{i}"])
                writer.writerow(row_to_write)

            with open(csv_total_file_for_16, "a") as csvtotalfile:
                writer = csv.DictWriter(
                    csvtotalfile, fieldnames=csv_total_columns_for_16
                )
                row_to_write = {}
                row_to_write["exp_name"] = exp_name
                row_to_write["for_16"] = np.mean(results_for_16["for_16"])
                for i in range(num_views):
                    row_to_write[f"for_16_{i}"] = np.mean(results_for_16[f"for_16_{i}"])
                writer.writerow(row_to_write)
