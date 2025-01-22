import os
from PIL import Image

# Define the folder paths
folders = ["DA2", "lotus", "depth-pro"]
output_folder = "concatted"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Collect all image files and group by name (without extension)
image_groups = {}
for folder in folders:
    for file in os.listdir(folder):
        if file.endswith(".png") or file.endswith(".jpg"):
            name_without_ext = os.path.splitext(file)[0]
            image_path = os.path.join(folder, file)
            image_groups.setdefault(name_without_ext, []).append(image_path)

# Function to create a 2x2 grid of images with specific layout
def create_grid(images):
    # Open images from DA2 (first row) and lotus/depth-pro (second row)
    first_row_image = Image.open(images[0])  # DA2
    second_row_images = [Image.open(img) for img in images[1:3]]  # lotus and depth-pro

    # Determine the maximum width and height for resizing
    max_width = max(first_row_image.width // 2, max(img.width for img in second_row_images))
    max_height = max(first_row_image.height // 2, max(img.height for img in second_row_images))
    uniform_size = (max_width, max_height)

    # Resize the first row image to span two slots horizontally
    first_row_resized = first_row_image.resize((2 * max_width, max_height), Image.ANTIALIAS)

    # Resize all second row images to the same size
    resized_second_row = [img.resize(uniform_size, Image.ANTIALIAS) for img in second_row_images]

    # Create a blank canvas for the grid
    grid_image = Image.new("RGB", (2 * max_width, 2 * max_height), "white")

    # Paste images onto the grid
    grid_image.paste(first_row_resized, (0, 0))  # First row spans two slots
    positions = [(0, max_height), (max_width, max_height)]  # Second row positions
    for img, pos in zip(resized_second_row, positions):
        grid_image.paste(img, pos)

    return grid_image

# Process each group of images and save the combined result
for name, image_paths in image_groups.items():
    if len(image_paths) > 2:  # Only process if there are matching names
        # Take images for the grid in the specified layout
        images_for_grid = [image_paths[0]] + image_paths[1:3]  # First from DA2, then lotus and depth-pro

        # Create the grid image
        grid = create_grid(images_for_grid)

        # Save the grid image to the output folder
        output_path = os.path.join(output_folder, f"{name}_combined.jpg")
        grid.save(output_path)
        print(f"Saved: {output_path}")

print("Processing complete.")
