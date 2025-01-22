#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""


import argparse
import logging
from pathlib import Path


import numpy as np
import PIL.Image
from PIL import ImageDraw, ImageFont
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
from depth_pro import create_model_and_transforms, load_rgb

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

import json
def run(args):
    """Run Depth Pro on a sample image."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Load model.
    model, transform = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    trues = 0
    falses = 0

    with open('/home/arda/OGAM/DA-2K/DA-2K/annotations.json', 'r') as f:
        annotations = json.load(f)
        first_10_items = list(annotations.items())[:200]

    for image_path, pairs in first_10_items:
        image_path_list = image_path.split("/")

        image_path_ = "/home/arda/OGAM/DA-2K/DA-2K/" + image_path
        point1_coordinates = pairs[0]['point1']
        point2_coordinates = pairs[0]['point2']
        print(point1_coordinates," ",point2_coordinates)
        print(f'Progress : {image_path_}')
        # Load image and focal length from exif info (if found.).
        try:
            LOGGER.info(f"Loading image {image_path_} ...")
            image, _, f_px = load_rgb(image_path_)
        except Exception as e:
            LOGGER.error(str(e))
            continue
        # Run prediction. If `f_px` is provided, it is used to estimate the final metric depth,
        # otherwise the model estimates `f_px` to compute the depth metricness.
        torch.cuda.synchronize()
        start = time.time()

        #with torch.autograd.profiler.profile(use_cuda=True) as prof:
        temp = transform(image) # 2.7ms (probable pre-processing)
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
        prediction = model.infer(temp, f_px=f_px) 
        torch.cuda.synchronize()
        
    
        

        end = time.time()
        print(f"Execution time: {end - start:.6f} seconds")
        # Extract the depth and focal length.
        depth = prediction["depth"].detach().cpu().numpy().squeeze()

        
        if f_px is not None:
            LOGGER.debug(f"Focal length (from exif): {f_px:0.2f}")
        elif prediction["focallength_px"] is not None:
            focallength_px = prediction["focallength_px"].detach().cpu().item()
            LOGGER.info(f"Estimated focal length: {focallength_px}")

        inverse_depth = 1 / depth
        # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
        max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
        min_invdepth_vizu = max(1 / 250, inverse_depth.min())
        inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
            max_invdepth_vizu - min_invdepth_vizu
        )

        point1_depth = inverse_depth[point1_coordinates[0]][point1_coordinates[1]] #WRONG RIGHT CLASSIFICATION AMBLEM
        point2_depth = inverse_depth[point2_coordinates[0]][point2_coordinates[1]]
        print(point1_depth," ",point2_depth," ",point1_depth-point2_depth)
        if point1_depth>point2_depth:
            print("HELL YEAH! POINTWISE CORRECT!")
            trues+=1
        else:
            print("YIKES! WE GOT A PROBLEM HERE!")
            falses+=1

        # Save Depth as npz file.
        if args.output_path is not None:
            output_file = (
                args.output_path
                
            )
            LOGGER.info(f"Saving depth map to: {str(output_file)}")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_file, depth=depth)

            # Save as color-mapped "turbo" jpg image.
            cmap = plt.get_cmap("turbo")
            color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
                np.uint8
            )
            color_map_output_file = "/home/arda/OGAM/ml-depth-pro/depths/"+image_path_list[-1]
            print("Output file:", image_path)
            print("Cmap_output_file:",color_map_output_file)
            LOGGER.info(f"Saving color-mapped depth to: : {color_map_output_file}")
            image = PIL.Image.fromarray(color_depth)
            # Create a drawing object
            draw = ImageDraw.Draw(image)

            # Add dots (circles) at specific positions
            dot_positions = [(100, 100), (300, 300)]  # List of (x, y) positions
            dot_radius = 10  # Radius of the dots

            
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
            font = ImageFont.truetype(font_path, size=60)#changed
            # Add text
            if point1_depth > point2_depth:
                draw.text((50, 50), "Correct Classification", fill="white", font=font)
                print("HEELLL YEEAHHH")
            else:
                draw.text((50,50), "Wrong Classification", fill="red", font=font)  # Add green text
                print("YIKES!!!")

            # Save the image as a JPEG file
            image.save(color_map_output_file, format="JPEG", quality=90)
            print("done it here pal")

        # Display the image and estimated depth map.
        #if not args.skip_display:
        #    ax_rgb.imshow(image)
        #    ax_disp.imshow(inverse_depth_normalized, cmap="turbo")
        #    fig.canvas.draw()
        #    fig.canvas.flush_events()

    LOGGER.info("Done predicting depth!")
    if not args.skip_display:
        plt.show(block=True)
    print("OVERALL ACCURACY:")
    print(str(trues/(trues+falses)))


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/example.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Skip matplotlib display.",
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show verbose output."
    )
    
    run(parser.parse_args())


if __name__ == "__main__":
    main()
