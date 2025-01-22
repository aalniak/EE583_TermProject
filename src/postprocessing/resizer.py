from PIL import Image
import os

def resize_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.jpg'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            try:
                with Image.open(input_path) as img:
                    width, height = img.size
                    if width > height:
                        new_size = (1280, 720)
                    else:
                        new_size = (720, 1280)
                    
                    img_resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)
                    img_resized.save(output_path)
                    print(f"Processed {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Update these paths accordingly
input_folder = "."
output_folder = "."

resize_images(input_folder, output_folder)
