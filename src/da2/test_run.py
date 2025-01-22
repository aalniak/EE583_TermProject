import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import json
import time
from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    #parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    #if os.path.isfile(args.img_path):
    #    if args.img_path.endswith('txt'):
    #        with open(args.img_path, 'r') as f:
    #            filenames = f.read().splitlines()
    #    else:
    #        filenames = [args.img_path]
    #else:
    #    filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    trues = 0
    falses = 0
    print("ASFSDGA")
    with open('/home/arda/OGAM/DA-2K/DA-2K/annotations.json', 'r') as f:
        annotations = json.load(f)
        first_50_items = list(annotations.items())[:500]
    for image_path, pairs in first_50_items:
        image_path = "/home/arda/OGAM/DA-2K/DA-2K/" + image_path
        point1_coordinates = pairs[0]['point1']
        point2_coordinates = pairs[0]['point2']
        print(point1_coordinates," ",point2_coordinates)
        print(f'Progress : {image_path}')
        
        raw_image = cv2.imread(image_path)
        start_time = time.time()
        depth = depth_anything.infer_image(raw_image, args.input_size)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.6f} seconds")
        npz_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(image_path))[0] + '.npz')
        np.savez_compressed(npz_path, depth=depth)
        print(f"Depth map saved as {npz_path}")
        start_time = time.time()
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        point1_depth = depth[point1_coordinates[0]][point1_coordinates[1]] #WRONG RIGHT CLASSIFICATION AMBLEM
        point2_depth = depth[point2_coordinates[0]][point2_coordinates[1]]

        if point1_depth>point2_depth:
            print("HELL YEAH! POINTWISE CORRECT!")
            trues+=1
        else:
            print("YIKES! WE GOT A PROBLEM HERE!")
            falses+=1
        print(f"NumPy Execution time: {start_time - end_time:.6f} seconds")
        if args.grayscale:
            depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        cv2.circle(depth, (point1_coordinates[1],point1_coordinates[0]), radius=10, color=(0, 255, 0), thickness=-1)  # Blue dot
        cv2.circle(depth, (point2_coordinates[1],point2_coordinates[0]), radius=10, color=(0, 0, 255), thickness=-1)  # Red dot

        if point1_depth > point2_depth:
            text = "Correct Classification"
        else:
            text = "Wrong Classification"
        
        cv2.putText(
        depth, 
        text, 
        (50,100), 
        cv2.FONT_HERSHEY_COMPLEX, 
        fontScale=2,  # Font size
        color=(255, 255, 255),  # Green color (BGR)
        thickness=2, 
        lineType=cv2.LINE_AA
        )

        if args.pred_only:
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(image_path))[0] + '.png'), depth)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            combined_result = cv2.hconcat([raw_image, split_region, depth])
            
            cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(image_path))[0] + '.png'), combined_result)
    print("OVERALL ACCURACY:")
    print(str(trues/(trues+falses)))