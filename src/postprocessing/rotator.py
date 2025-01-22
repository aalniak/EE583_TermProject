from PIL import Image, ImageOps
import os

def rotate_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            try:
                with Image.open(input_path) as img:
                    # Correct any unwanted rotations due to EXIF metadata
                    img = ImageOps.exif_transpose(img)

                    # Display the image for review
                    img.show()
                    print(f"Currently viewing: {filename}")

                    # Ask the user for rotation
                    rotate = input("Rotate the image? (y/n): ").strip().lower()
                    if rotate == 'y':
                        angle = int(input("Enter the rotation angle (e.g., 90, -90, 180): ").strip())
                        img = img.rotate(angle, expand=True)

                    img.save(output_path)
                    print(f"Processed and saved: {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

# Update these paths accordingly
input_folder = "."
output_folder = "."

rotate_images(input_folder, output_folder)
