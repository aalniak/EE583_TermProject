import logging
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from contextlib import nullcontext
import time
import numpy as np
import torch
import cProfile
import pstats
from tqdm.auto import tqdm
from diffusers.utils import check_min_version

from pipeline import LotusGPipeline, LotusDPipeline
from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all

check_min_version('0.28.0.dev0')

def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run Lotus..."
    )
    # model settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The used prediction_type. ",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="regression", # "generation"
        help="Whether to use the generation or regression pipeline."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth", # "normal"
    )
    parser.add_argument(
        "--disparity",
        action="store_true",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # inference settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    args = parser.parse_args()

    return args
import cv2
import json
def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run inference...")

    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Output dir = {args.output_dir}")

    output_dir_color = os.path.join(args.output_dir, f'{args.task_name}_vis')
    output_dir_npy = os.path.join(args.output_dir, f'{args.task_name}')
    if not os.path.exists(output_dir_color): os.makedirs(output_dir_color)
    if not os.path.exists(output_dir_npy): os.makedirs(output_dir_npy)

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Data --------------------
    root_dir = Path(args.input_dir)
    test_images = list(root_dir.rglob('*.png')) + list(root_dir.rglob('*.jpg'))
    test_images = sorted(test_images)
    print('==> There are', len(test_images), 'images for validation.')
    # -------------------- Model --------------------

    if args.mode == 'generation':
        pipeline = LotusGPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    elif args.mode == 'regression':
        pipeline = LotusDPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            torch_dtype=dtype,
        )
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    logging.info(f"Successfully loading pipeline from {args.pretrained_model_name_or_path}.")

    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        with open('/home/arda/OGAM/DA-2K/DA-2K/annotations.json', 'r') as f:
            annotations = json.load(f)
            first_10_items = list(annotations.items())[:200]
        trues = 0
        falses = 0
        for image_path, pairs in first_10_items:
            image_path_list = image_path.split("/")

            image_path_ = "/home/arda/OGAM/DA-2K/DA-2K/" + image_path
            point1_coordinates = [pairs[0]['point1'][0]//2,pairs[0]['point1'][1]//2]
            point2_coordinates = [pairs[0]['point2'][0]//2,pairs[0]['point2'][1]//2]
            print(point1_coordinates," ",point2_coordinates)
            print(f'Progress : {image_path_}')

            if torch.backends.mps.is_available():
                autocast_ctx = nullcontext()
            else:
                autocast_ctx = torch.autocast(pipeline.device.type)
            with autocast_ctx:
                # Preprocess validation image
                #with torch.autograd.profiler.profile(use_cuda=True) as prof:
                start = time.time()
                test_image = Image.open(image_path_).convert('RGB')
                test_image = np.array(test_image).astype(np.float32)
                height, width = test_image.shape[:2]  # Get original dimensions
                new_dimensions = (int(width * 0.5), int(height * 0.5))  # Compute new dimensions
                test_image = cv2.resize(test_image, new_dimensions, interpolation=cv2.INTER_AREA)
                
                

                test_image = torch.tensor(test_image).permute(2,0,1).unsqueeze(0)
                test_image = test_image / 127.5 - 1.0 
                test_image = test_image.to(device)

                task_emb = torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(device)
                task_emb = torch.cat([torch.sin(task_emb), torch.cos(task_emb)], dim=-1).repeat(1, 1)
                end = time.time()
                inference_time = end - start
                print(f"Total preprocessing time for {image_path} is {inference_time:.4f} seconds")
                #print(prof.key_averages().table(sort_by="cuda_time_total"))
                torch.cuda.synchronize()
                start = time.time()
                # Run
                #profiler = cProfile.Profile()
                #profiler.enable()
                pred = pipeline.__call__(
                    rgb_in=test_image, 
                    prompt='', 
                    num_inference_steps=1, 
                    generator=generator, 
                    # guidance_scale=0,
                    output_type='np',
                    timesteps=[args.timestep],
                    task_emb=task_emb,
                    
                    ).images[0]
                #profiler.disable()
                #stats = pstats.Stats(profiler)
                #stats.sort_stats('cumulative').print_stats(20)
                torch.cuda.synchronize()
                end = time.time()
                inference_time = end - start
                print(f"Total pred pipeline time for {image_path} is {inference_time:.4f} seconds")
               
                # Post-process the prediction
                save_file_name = "/home/arda/OGAM/Lotus/depths/"+image_path_list[-1]
                if args.task_name == 'depth':
                    output_npy = pred.mean(axis=-1)
                    output_color = colorize_depth_map(output_npy, reverse_color=args.disparity)
                else:
                    output_npy = pred
                    output_color = Image.fromarray((output_npy * 255).astype(np.uint8))

                point1_depth = output_npy[point1_coordinates[0]][point1_coordinates[1]] #WRONG RIGHT CLASSIFICATION AMBLEM
                point2_depth = output_npy[point2_coordinates[0]][point2_coordinates[1]]

                print(point1_depth," ",point2_depth," ",point1_depth-point2_depth)
                if point1_depth>point2_depth:
                    print("HELL YEAH! POINTWISE CORRECT!")
                    trues+=1
                else:
                    print("YIKES! WE GOT A PROBLEM HERE!")
                    falses+=1
                dot_radius = 5  # Radius of the dots
                draw = ImageDraw.Draw(output_color)

                draw.ellipse(
                [point1_coordinates[1] - dot_radius, point1_coordinates[0] - dot_radius, point1_coordinates[1] + dot_radius, point1_coordinates[0] + dot_radius],
                fill=(0,255,0),  # Color of the dot
                outline=(0,0,0)  # Border color (optional)
                )

                draw.ellipse(
                    [point2_coordinates[1] - dot_radius, point2_coordinates[0] - dot_radius, point2_coordinates[1] + dot_radius, point2_coordinates[0] + dot_radius],
                    fill="red",  # Color of the dot
                    outline="black"  # Border color (optional)
                )

                font_path = "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
                font = ImageFont.truetype(font_path, size=30)

                if point1_depth > point2_depth:
                    draw.text((25, 25), "Correct Classification", fill="white", font=font)
                    print("HEELLL YEEAHHH")
                else:
                    draw.text((25,25), "Wrong Classification", fill="red", font=font)  # Add green text
                    print("YIKES!!!")
                print(output_color)
                print(output_dir_color)
                print(os.path.join(output_dir_color))
                output_color.save(save_file_name)
                np.save(os.path.join(save_file_name[:-4]+'.npy'), output_npy)

            torch.cuda.empty_cache()
            end = time.time()
            inference_time = end - start
            print(f"Total inference time for {image_path} is {inference_time:.4f} seconds")
            print(" ")

    print('==> Inference is done. \n==> Results saved to:', args.output_dir)
    print("OVERALL ACCURACY:",str(trues/(trues+falses)))


if __name__ == '__main__':
    main()
