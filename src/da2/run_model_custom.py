import cv2
import torch
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    #'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

raw_img = cv2.imread('../jpgs/dashcam.jpg')
if raw_img is None:
    print("Error: Image not found. Please ensure the path to the image is correct.")
    exit()

inference_times = {}


for encoder, config in model_configs.items():
    print(f"\nTesting model: {encoder.upper()}")

    # Initialize and load model
    model = DepthAnythingV2(**config)
    model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
    model = model.to(DEVICE).eval()
    depth = model.infer_image(raw_img)
    # Perform inference and measure time
    start_time = time.time()
    #with torch.autograd.profiler.profile(use_cuda=True) as prof:
    depth = model.infer_image(raw_img)
    #print(prof.key_averages().table(sort_by="cuda_time_total"))
    end_time = time.time()

    inference_time = end_time - start_time
    inference_times[encoder] = inference_time

    print(f"Inference Time for {encoder.upper()}: {inference_time:.4f} seconds")

    # Save depth map visualization
    plt.imshow(depth, cmap='hot')
    plt.axis('off')
    plt.title(f"Depth Map ({encoder.upper()})")
    output_filename = f"depth_dashcam_{encoder}.png"
    plt.savefig(output_filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Depth map saved as {output_filename}")

# Print summary of inference times
print("\nSummary of Inference Times:")
for encoder, time_taken in inference_times.items():
    print(f"Model {encoder.upper()}: {time_taken:.4f} seconds")

