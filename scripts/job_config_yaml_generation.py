import yaml
import csv
from vivid123.configs import ViVid123BaseSchema


my_model = ViVid123BaseSchema()

with open("scripts/gso_metadata_object_prompt_100.csv", 'r') as f_metadata:
    csv_lines = csv.reader(f_metadata, delimiter=',', quotechar='"')
    for i, csv_line in enumerate(csv_lines):
        obj_name = csv_line[0]
        my_model.name = obj_name
        my_model.input_image_path = f"tmp/{obj_name}/img/012.png"
        with open(f"tasks/{obj_name}.yaml", "w") as f_job:
            yaml.dump(my_model.model_dump(), f_job)
