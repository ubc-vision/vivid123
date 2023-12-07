# vivid123

## Requirements
```bash
pip install torch "diffusers>0.23" transformers accelerate einops kornia imageio[ffmpeg] opencv pydantic
```

## Run single generation task
```bash
python run_generation.py
```

## Prepare batch generation config yaml file
```bash
python -m scripts.job_config_yaml_generation 
```
This will put all the yaml files in the `tasks` folder.

## Run batch generation tasks
```bash
CUDA_VISIBLE_DEVICES=0 python run_batch_generation.py --task_yamls_dir=tasks --dataset_dir=gso-rendered-reference-45-starting-0-ending-90 --output_dir=outputs --obj_csv_file=scripts/gso_metadata_object_prompt_100.csv --run_from_obj_index=0 --run_to_obj_index=50
```

## Tips for scheduling batch generation on SLURM clusters
It takes about 1min30s to run one generation on a v100 gpu. If the number of generation is too large for each job you can scheudle on a SLURM cluster, 
you can split dataset for each job using the `--run_from_obj_index` and `--run_to_obj_index` options.

TODO: ipython notebook