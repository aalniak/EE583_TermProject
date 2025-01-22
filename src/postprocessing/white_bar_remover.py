import os
from PIL import Image

# Define the folder paths
input_folder = "."  # Current folder
output_folder = "dump"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for file in os.listdir(input_folder):
    if file.endswith(".png") or file.endswith(".jpg"):
        input_path = os.path.join(input_folder, file)

        # Open the image
        with Image.open(input_path) as img:
            width, height = img.size

            # Calculate the crop regions
            left_part = img.crop((0, 0, (width // 2) - 25, height))  # Left part up to the white bar
            right_part = img.crop(((width // 2) + 25, 0, width, height))  # Right part after the white bar

            # Create a new image without the white bar
            new_width = left_part.width + right_part.width
            new_image = Image.new("RGB", (new_width, height))

            # Paste the two parts together
            new_image.paste(left_part, (0, 0))
            new_image.paste(right_part, (left_part.width, 0))

            # Ensure the output image has the correct dimensions
            new_image = new_image.crop((0, 0, new_width, height))

            # Save the new image to the output folder
            output_path = os.path.join(output_folder, file)
            new_image.save(output_path)
            print(f"Processed and saved: {output_path}")

print("Processing complete.")
