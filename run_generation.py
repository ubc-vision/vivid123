import argparse

from vivid123 import generation_vivid123, prepare_vivid123_pipeline

ZERO123_MODEL_ID = "bennyguo/zero123-xl-diffusers"
VIDEO_MODEL_ID = "cerspense/zeroscope_v2_576w"
VIDEO_XL_MODEL_ID = "cerspense/zeroscope_v2_XL"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViVid123 Generation')
    parser.add_argument('--task_yaml_path', type=str, required=True, help='The path for the task yaml')
    args = parser.parse_args()
    
    vivid123_pipe, xl_pipe = prepare_vivid123_pipeline(
        ZERO123_MODEL_ID=ZERO123_MODEL_ID, 
        VIDEO_MODEL_ID=VIDEO_MODEL_ID, 
        VIDEO_XL_MODEL_ID=VIDEO_XL_MODEL_ID
    )

    generation_vivid123(config_path=args.task_yaml_path, vivid123_pipe=vivid123_pipe, xl_pipe=xl_pipe)