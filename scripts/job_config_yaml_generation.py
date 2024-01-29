import os
import shutil
import yaml
import csv
import argparse
from vivid123.configs import ViVid123BaseSchema


SLURM_TMPDIR = os.getenv("SLURM_TMPDIR") if os.getenv("SLURM_TMPDIR") else "/home/erqun/vivid123/tmp"

job_specs = [
    # {"num_frames": 24, "delta_azimuth_start": 15, "delta_azimuth_end": 360, "exp_name": "num_frames_24"},
    {}  # default job specified by default schema in vivid123/configs/base_schema.py
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViVid123 Generation')
    parser.add_argument('--run_on_slurm', action='store_true', help="whether to run on a slurm cluster")
    args = parser.parse_args()

    for job_spec in job_specs:
        with open("scripts/gso_metadata_object_prompt_100.csv", 'r') as f_metadata:
            csv_lines = csv.reader(f_metadata, delimiter=',', quotechar='"')
            my_model = ViVid123BaseSchema()
            for fieldname, value in job_spec.items():
                if hasattr(my_model, fieldname):
                    setattr(my_model, fieldname, value)
                else:
                    raise ValueError(f"No field {fieldname}")

            task_yamls_output_dir = f"exps/task_yamls/{my_model.exp_name}"
            os.makedirs(task_yamls_output_dir, exist_ok=True)
            for i, csv_line in enumerate(csv_lines):
                my_model.obj_name = csv_line[0]
                if args.run_on_slurm:
                    my_model.input_image_path = r"${SLURM_TMPDIR}/" + f"{my_model.obj_name}/img/012.png"
                else:
                    my_model.input_image_path = f"./tmp/{my_model.obj_name}/img/012.png"
                with open(os.path.join(task_yamls_output_dir, f"{my_model.obj_name}.yaml"), "w") as f_job:
                    print(f"dumping yaml to ", os.path.join(task_yamls_output_dir, f"{my_model.obj_name}.yaml"))
                    yaml.dump(my_model.model_dump(), f_job)
