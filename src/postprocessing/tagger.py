import os
import cv2

# Define the folder containing the images
input_folder = "."  # Current folder
output_folder = "output_with_text"

# Text to write on the images
text_to_write = "Sample Text"
font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
font_scale = 1.5  # Font scale
font_color = (255, 255, 255)  # White color
font_thickness = 3  # Thickness of the text
border_color = (0, 0, 0)  # Black color for the border
border_thickness = 4 # Thickness of the border

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process each image in the folder
for file in os.listdir(input_folder):
    if file.endswith(".png") or file.endswith(".jpg"):
        input_path = os.path.join(input_folder, file)

        # Read the image
        image = cv2.imread(input_path)
        if image is None:
            print(f"Failed to read {input_path}, skipping.")
            continue

        # Get image dimensions
        height, width, _ = image.shape

        # Position for the text (bottom center of the image)
        text_size = cv2.getTextSize("Original Image", font, font_scale, font_thickness)[0]
        text_x = width//2-text_size[0]-30  # 1
        text_y = height//2 - 30  # 1

        # Write the text on the image
        cv2.putText(image, "Original Image", (text_x, text_y), font, font_scale, border_color, border_thickness)
        cv2.putText(image, "Original Image", (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Position for the text (bottom center of the image)
        text_size = cv2.getTextSize("Depth-Anything-V2", font, font_scale, font_thickness)[0]
        text_x = width-text_size[0]-30  # Center horizontally
        text_y = height//2 - 30  # 20 pixels from the bottom

        # Write the text on the image
        cv2.putText(image, "Depth-Anything-V2", (text_x, text_y), font, font_scale, border_color, border_thickness)
        cv2.putText(image, "Depth-Anything-V2", (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Position for the text (bottom center of the image)
        text_size = cv2.getTextSize('Lotus', font, font_scale, font_thickness)[0]
        text_x = width//2-text_size[0]-30  # Center horizontally
        text_y = height - 30  # 20 pixels from the bottom

        # Write the text on the image
        cv2.putText(image, 'Lotus', (text_x, text_y), font, font_scale, border_color, border_thickness)
        cv2.putText(image, 'Lotus', (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Position for the text (bottom center of the image)
        text_size = cv2.getTextSize("DepthPro", font, font_scale, font_thickness)[0]
        text_x = width-text_size[0]-30  # Center horizontally
        text_y = height - 30  # 20 pixels from the bottom

        # Write the text on the image
        cv2.putText(image, "DepthPro", (text_x, text_y), font, font_scale, border_color, border_thickness)
        cv2.putText(image, "DepthPro", (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Save the image with the text
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, image)
        print(f"Processed and saved: {output_path}")

print("Processing complete.")
