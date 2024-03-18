# ViVid-1-to-3: Novel View Synthesis with Video Diffusion Models

This repository is a reference implementation for ViVid-1-to-3. It combines video diffusion with novel-view synthesis diffusion models for increased pose and appearace consistency.

[[arXiv]](https://arxiv.org/abs/2312.01305), [[project page]](https://ubc-vision.github.io/vivid123/)

## Requirements
```bash
pip install torch "diffusers>0.23" transformers accelerate einops kornia imageio[ffmpeg] opencv-python pydantic scikit-image lpips
```

## Run single generation task
Put the reference image to $IMAGE_PATH, and set the `input_image_path` in `scripts/task_example.yaml` to it. Then run
```bash
python run_generation.py --task_yaml_path=scripts/task_example.yaml
```

## Run batch generation tasks
We have supported running batch generation tasks on both PC and SLURM clusters.
### Prepare batch generation config yaml file
We tested our method on 100 [GSO](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research) objects. The list of the objects is in `scripts/gso_metadata_object_prompt_100.csv`, along with our labeled text prompts if you would like to test prompt-based generation yourself. We have rendered the 100 objects beforehand. It can be downloaded [here](https://drive.google.com/file/d/1A9PJDRD27igX5p88slWVF_QSDKxaZDCZ/view?usp=sharing). You can decompress the content into `gso-100`. Then simply run the following line to prepare a batch generation job on a PC:
```bash
python -m scripts.job_config_yaml_generation 
```
. Or run the following line to prepare a batch generation job on a SLURM cluster:
```
python -m scripts.job_config_yaml_generation --run_on_slurm
```
All the yaml files will be generated in a new folder called `tasks_gso`.

### Batch generation
```bash
python run_batch_generation.py --task_yamls_dir=tasks_gso --dataset_dir=gso-100 --output_dir=outputs --obj_csv_file=scripts/gso_metadata_object_prompt_100.csv
```

### Tips for scheduling batch generation on SLURM clusters
It takes about 1min30s to run one generation on a v100 gpu. If the number of generations is too large for each job you can schedule on a SLURM cluster, 
you can split the dataset for each job using the `--run_from_obj_index` and `--run_to_obj_index` options. For example
```bash
python run_batch_generation.py --task_yamls_dir=tasks_gso --dataset_dir=gso-100 --output_dir=outputs --obj_csv_file=scripts/gso_metadata_object_prompt_100.csv --run_from_obj_index=0 --run_to_obj_index=50
```

## TODO
- [ ] Evaluation code
- [ ] iPython notebook

## Acknowledgement
This repo is based on the Huggingface community [implementation](https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_zero1to3.py) and [converted weights](https://huggingface.co/bennyguo/zero123-xl-diffusers) of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123), as well as the Huggingface community text-to-video model [Zeroscope v2](https://huggingface.co/cerspense/zeroscope_v2_576w). Thanks for their awesome works.

## Citation

If you use this code in your research, please cite our paper:
```
@article{kwak2023vivid,
  title={ViVid-1-to-3: Novel View Synthesis with Video Diffusion Models},
  author={Kwak, Jeong-gi and Dong, Erqun and Jin, Yuhe and Ko, Hanseok and Mahajan, Shweta and Yi, Kwang Moo},
  journal={arXiv preprint arXiv:2312.01305},
  year={2023}
}
```
