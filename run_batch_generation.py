import os
import argparse
import shutil
import csv

from vivid123 import generation_vivid123, prepare_pipelines


ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
VIDEO_MODEL_ID = "cerspense/zeroscope_v2_576w"
VIDEO_XL_MODEL_ID = "cerspense/zeroscope_v2_XL"

SLURM_TMPDIR = os.getenv("SLURM_TMPDIR") if os.getenv("SLURM_TMPDIR") else "./tmp"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViVid123 Generation')
    parser.add_argument('--task_yamls_dir', type=str, required=True, help='The directory for all configs')
    parser.add_argument('--dataset_dir', type=str, required=True, help='The directory for all groundtruth renderings, each object being a zip file')
    parser.add_argument('--output_dir', type=str, required=True, help='The root directory for all outputs')
    parser.add_argument('--obj_csv_file', type=str, required=True, help='The csv file for all objects')
    parser.add_argument('--run_from_obj_index', type=int, default=0, help='The index of object to start with')
    parser.add_argument('--run_to_obj_index', type=int, default=999, help='The index of object to end with')
    args = parser.parse_args()
    
    vivid123_pipe, xl_pipe = prepare_pipelines(
        ZERO123_MODEL_ID=ZERO123_MODEL_ID, 
        VIDEO_MODEL_ID=VIDEO_MODEL_ID, 
        VIDEO_XL_MODEL_ID=VIDEO_XL_MODEL_ID
    )

    with open(args.obj_csv_file, 'r') as csv_file:
        csv_lines = csv.reader(csv_file, delimiter=',', quotechar='"')
        for i, csv_line in enumerate(csv_lines):
            if i < args.run_from_obj_index:
                continue
            if i > args.run_to_obj_index:
                break
            
            obj_name = csv_line[0]
            if os.path.isfile(f"{args.output_dir}/{obj_name}/xl.mp4"):
                print(f"{obj_name} has already been generated, skipping...")
                continue

            print(f"Processing {obj_name}")
            if not os.path.exists(f"{SLURM_TMPDIR}/{obj_name}"):
                print(f"unpacking {args.dataset_dir}/{obj_name}.zip to {SLURM_TMPDIR}/{obj_name}")
                shutil.unpack_archive(f"{args.dataset_dir}/{obj_name}.zip", f"{SLURM_TMPDIR}/{obj_name}")

            generation_vivid123(
                vivid123_pipe=vivid123_pipe, 
                xl_pipe=xl_pipe,
                config_path=f"{args.task_yamls_dir}/{obj_name}.yaml",
                output_root_dir=args.output_dir,
            )