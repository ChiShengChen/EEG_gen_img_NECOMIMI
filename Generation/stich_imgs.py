import os
import glob
from PIL import Image

# Total number of composite images to generate
num_composite_images = 20

# Size of each small image
desired_width = 128
desired_height = 128

# Maximum value of XXX (because the last group is 191~199)
max_XXX = 200

# Create the directory to save the composite images
output_dir = "/home/aidan/EEG_Image_decode_old/Generation/final_gen_stiched_img"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Directory created: {output_dir}")

# Generate each composite image
for composite_index in range(num_composite_images):
    # Calculate the range of XXX for the current group
    start_XXX = composite_index * 10 + 1
    end_XXX = start_XXX + 9

    # Special handling for the last group to prevent exceeding max_XXX
    if end_XXX > max_XXX:
        end_XXX = max_XXX

    XXX_values_first_row = list(range(start_XXX, end_XXX + 1))  # First row numbers start from 1
    XXX_values_other_rows = list(range(start_XXX - 1, end_XXX))  # Numbers for rows 2 to 4 start from 0


    # Store the images for each row
    row_images = [[] for _ in range(4)]  # 4 columns

    model_prefix = ""  # Used to store the "subK" extracted from the second row of images

    for i, XXX in enumerate(XXX_values_other_rows):
        # First row images - use the numbers from the first row (starting from 1)
        XXX_str_row1 = str(XXX_values_first_row[i]).zfill(5)

        # Find the folder starting with XXX_str_row1, then read the image from the folder
        path_row1_base = "/home/aidan/data/thingseeg/image_set/test_images"
        folder_pattern_row1 = os.path.join(path_row1_base, f"{XXX_str_row1}_*")
        folder_row1 = glob.glob(folder_pattern_row1)
        if not folder_row1:
            print(f"Folder not found: {folder_pattern_row1}")
            continue

        # Assume there is only one matching folder for each number
        folder_path_row1 = folder_row1[0]

        # Find the image in the folder
        image_pattern_row1 = os.path.join(folder_path_row1, "*")
        image_files_row1 = glob.glob(image_pattern_row1)
        if not image_files_row1:
            print(f"No image found in folder: {folder_path_row1}")
            continue

        image_path_row1 = image_files_row1[0]
        print(image_path_row1)
        image_row1 = Image.open(image_path_row1).resize((desired_width, desired_height))
        row_images[0].append(image_row1)

        # Second row images - use non-zero-padded numbers (starting from 0, no zero padding)
        XXX_str = str(XXX)  # Directly use the number, without zero padding
        mdl = "Nerv2"
        sub = "sub-02"
        seed = "_5757"
        # seed = ""
        # path_row2 = f"/home/aidan/EEG_Image_decode_old/Generation/gen_images/NICE_sub10_generated_image_{XXX_str}_from_image_embeds.png"
        path_row2 = f"/home/aidan/EEG_Image_decode_old/Generation/gen_images/{mdl}_{sub}_generated_image_{XXX_str}_from_image_embeds{seed}.png"

        print(path_row2)
        if not os.path.exists(path_row2):
            print(f"File not found: {path_row2}")
            continue
        image_row2 = Image.open(path_row2).resize((desired_width, desired_height))
        row_images[1].append(image_row2)

        

        # Third row images
        path_row3 = f"/home/aidan/EEG_Image_decode_old/Generation/gen_images/{mdl}_{sub}_generated_image_{XXX_str}_from_eeg_embeds{seed}.png"
        print(path_row3)
        if not os.path.exists(path_row3):
            print(f"File not found: {path_row3}")
            continue
        image_row3 = Image.open(path_row3).resize((desired_width, desired_height))
        row_images[2].append(image_row3)

        # Extract the prefix from the third row images (e.g., "NICE_sub10")
        if i == 0:  # Only need to extract from the first image
            model_prefix = os.path.basename(path_row3).split('_from')[0]

        # Fourth row images
        path_row4 = f"/home/aidan/EEG_Image_decode_old/Generation/gen_images/{mdl}_{sub}_generated_image_{XXX_str}_from_pipe_output{seed}.png"
        print(path_row4)
        if not os.path.exists(path_row4):
            print(f"File not found: {path_row4}")
            continue
        image_row4 = Image.open(path_row4).resize((desired_width, desired_height))
        row_images[3].append(image_row4)

    # Check if all rows have images
    if any(len(row) == 0 for row in row_images):
        print(f"Some rows are missing images while generating composite image {composite_index + 1}, skipping.")
        continue


    # Merge images in each row
    combined_rows = []
    for row in row_images:
        total_width = desired_width * len(row)
        combined_row = Image.new('RGB', (total_width, desired_height))
        x_offset = 0
        for im in row:
            combined_row.paste(im, (x_offset, 0))
            x_offset += desired_width
        combined_rows.append(combined_row)

    # Merge all rows to generate the final composite image
    total_height = desired_height * len(combined_rows)
    composite_image = Image.new('RGB', (desired_width * len(XXX_values_first_row), total_height))
    y_offset = 0
    for row_img in combined_rows:
        composite_image.paste(row_img, (0, y_offset))
        y_offset += desired_height

    # Generate the filename and save the composite image
    output_file_name = f"{model_prefix}_composite_image_{composite_index + 1}{seed}.png"
    output_file_path = os.path.join(output_dir, output_file_name)
    composite_image.save(output_file_path)
    print(f"Composite image {composite_index + 1} saved to {output_file_path}")
