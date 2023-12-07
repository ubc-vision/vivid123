import os
import yaml
import csv
import argparse
from vivid123.configs import ViVid123BaseSchema


parser = argparse.ArgumentParser(description='ViVid123 Generation')
parser.add_argument('--task_yamls_output_dir', type=str, default="tasks_gso", help='The directory for all configs')
parser.add_argument('--run_on_slurm', action='store_true', help="whether to run on a slurm cluster")
args = parser.parse_args()

my_model = ViVid123BaseSchema()

os.makedirs(args.task_yaml_output_dir, exist_ok=True)

with open("scripts/gso_metadata_object_prompt_100.csv", 'r') as f_metadata:
    csv_lines = csv.reader(f_metadata, delimiter=',', quotechar='"')
    for i, csv_line in enumerate(csv_lines):
        obj_name = csv_line[0]
        my_model.name = obj_name
        if args.run_on_slurm:
            my_model.input_image_path = r"${SLURM_TMPDIR}/" + f"{obj_name}/img/012.png"
        else:
            my_model.input_image_path = f"./tmp/{obj_name}/img/012.png"
        with open(f"{args.task_yaml_output_dir}/{obj_name}.yaml", "w") as f_job:
            yaml.dump(my_model.model_dump(), f_job)
