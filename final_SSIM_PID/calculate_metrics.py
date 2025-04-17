import os
import glob
import re
import warnings
import argparse # Added for command line arguments
import numpy as np
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
from skimage import io, img_as_ubyte

# --- Configuration ---
GEN_IMAGES_DIR = '/home/aidan/EEG_Image_decode_old/Generation/gen_images'
GT_IMAGES_DIR = '/home/aidan/EEG_Image_decode_old/image_set/test_images'
OUTPUT_DIR = '/home/aidan/EEG_Image_decode_old/final_SSIM_PID'
SSIM_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'ssim_results.csv')
FID_RESULTS_FILE = os.path.join(OUTPUT_DIR, 'fid_results.csv')
IMAGE_SIZE = (256, 256) # Resize images to this size for SSIM consistency

# Regex to parse generated image filenames
# Handles names like: Nerv2_sub-02_generated_image_18_from_image_embeds_5757_ip1.png
# Or: ATM_S_sub01_generated_image_0_from_eeg_embeds.png
# Assumes subject ID is always two digits
FILENAME_REGEX = re.compile(
    r"^(.*?)_sub-?(\d{2})_generated_image_(\d+)_from_(eeg_embeds|image_embeds|pipe_output)(?:_\S+)?\.png$"
)

# --- Helper Functions ---

def find_gt_image(index, gt_base_dir):
    """Finds the ground truth image path corresponding to the index."""
    try:
        # Format the directory prefix (e.g., 00006 for index 5)
        dir_prefix = f"{index + 1:05d}"
        # Find the directory starting with the prefix
        matching_dirs = glob.glob(os.path.join(gt_base_dir, f"{dir_prefix}_*"))

        if not matching_dirs:
            print(f"Warning: No directory found for index {index} (prefix {dir_prefix}) in {gt_base_dir}")
            return None
        if len(matching_dirs) > 1:
            print(f"Warning: Multiple directories found for index {index} (prefix {dir_prefix}): {matching_dirs}. Using the first one.")

        gt_dir = matching_dirs[0]
        # Find the jpg file inside the directory
        matching_files = glob.glob(os.path.join(gt_dir, "*.jpg"))
        if not matching_files:
            matching_files = glob.glob(os.path.join(gt_dir, "*.png")) # Fallback for png

        if not matching_files:
             print(f"Warning: No jpg or png image found in {gt_dir} for index {index}")
             return None
        if len(matching_files) > 1:
             print(f"Warning: Multiple images found in {gt_dir}: {matching_files}. Using the first one.")

        return matching_files[0]

    except Exception as e:
        print(f"Error finding GT image for index {index}: {e}")
        return None

def calculate_ssim_for_pair(gen_path, gt_path, target_size):
    """Calculates SSIM between two images."""
    try:
        with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=UserWarning) # Ignore warnings from skimage io
             gen_img = io.imread(gen_path)
             gt_img = io.imread(gt_path)

        # Convert to grayscale if they have 3 dimensions (RGB or RGBA)
        if gen_img.ndim == 3:
            gen_img = img_as_ubyte(io.imread(gen_path, as_gray=True))
        if gt_img.ndim == 3:
             gt_img = img_as_ubyte(io.imread(gt_path, as_gray=True))

        # Resize for consistency
        gen_img_resized = resize(gen_img, target_size, anti_aliasing=True)
        gt_img_resized = resize(gt_img, target_size, anti_aliasing=True)

        # Ensure images are in the range [0, 1] or specify data_range
        gen_img_resized = np.clip(gen_img_resized, 0, 1)
        gt_img_resized = np.clip(gt_img_resized, 0, 1)

        score = ssim(gen_img_resized, gt_img_resized, data_range=1.0)
        return score
    except Exception as e:
        print(f"Error calculating SSIM for {gen_path} and {gt_path}: {e}")
        return None

def run_fid(path1, path2):
    """Runs the pytorch-fid command and returns the FID score."""
    # Note: pytorch-fid requires paths to directories containing images.
    # If path1 contains paths to specific files, we might need to copy them to a temp dir.
    # However, the command line tool usually expects directories.
    command = [
        'python', '-m', 'pytorch_fid',
        path1, path2,
        '--device', 'cuda', # Assuming CUDA is available, change to 'cpu' if not
        '--batch-size', '50', # Explicitly set batch size
        '--num-workers', '0' # Explicitly set number of workers to 0
    ]
    print(f"Running FID command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        print(f"FID Output:\n{output}")
        # Extract FID score from the output (usually the last line)
        match = re.search(r"FID:\s+([\d\.]+)", output)
        if match:
            return float(match.group(1))
        else:
            print("Error: Could not parse FID score from output.")
            return None
    except FileNotFoundError:
        print("Error: 'python -m pytorch_fid' command not found. Make sure pytorch-fid is installed and accessible.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error running FID command: {e}")
        print(f"Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during FID calculation: {e}")
        return None

# --- Main Logic ---

def main(gen_images_dir, gt_images_dir, output_dir, ssim_results_filename, image_size_str):
    os.makedirs(output_dir, exist_ok=True)

    try:
        img_size = tuple(map(int, image_size_str.split('x')))
        if len(img_size) != 2:
            raise ValueError("Image size must be in format WIDTHxHEIGHT, e.g., 256x256")
    except ValueError as e:
        print(f"Error parsing image size '{image_size_str}': {e}")
        return

    all_gen_files = glob.glob(os.path.join(gen_images_dir, "*.png"))
    print(f"Found {len(all_gen_files)} generated images in {gen_images_dir}")

    ssim_data = []
    processed_count = 0 # Counter for progress

    for gen_path in all_gen_files:
        filename = os.path.basename(gen_path)
        match = FILENAME_REGEX.match(filename)

        if not match:
            if "generated_image" in filename:
                 print(f"Warning: Could not parse generated filename: {filename}")
            continue

        model, subject_num, index_str, emb_type = match.groups()
        subject = f"sub-{subject_num}"
        index = int(index_str)

        gt_path = find_gt_image(index, gt_images_dir)
        if not gt_path:
            continue

        ssim_score = calculate_ssim_for_pair(gen_path, gt_path, img_size)
        if ssim_score is not None:
            ssim_data.append({
                'model': model,
                'subject': subject,
                'image_index': index,
                'emb_type': emb_type,
                'generated_path': gen_path,
                'gt_path': gt_path,
                'ssim_score': ssim_score
            })

        processed_count += 1
        if processed_count % 1000 == 0:
             print(f"Processed {processed_count}/{len(all_gen_files)} images for SSIM...")

    # --- Save SSIM Results ---
    if ssim_data:
        ssim_df = pd.DataFrame(ssim_data)
        output_path = os.path.join(output_dir, ssim_results_filename)
        ssim_df.to_csv(output_path, index=False)
        print(f"\nSSIM results saved to {output_path}")
    else:
        print("\nNo SSIM scores were calculated.")

    print("\nSSIM calculation script finished.")

if __name__ == "__main__":
    # Define default paths relative to a potential project root or use absolute paths
    # These defaults might need adjustment based on typical usage scenarios
    DEFAULT_GEN_DIR = './Generation/gen_images'
    DEFAULT_GT_DIR = './image_set/test_images'
    DEFAULT_OUTPUT_DIR = '.'

    parser = argparse.ArgumentParser(description="Calculate SSIM scores between generated images and ground truth images.")
    parser.add_argument("--gen_images_dir", type=str, default=DEFAULT_GEN_DIR,
                        help=f"Directory containing generated images. Default: {DEFAULT_GEN_DIR}")
    parser.add_argument("--gt_images_dir", type=str, default=DEFAULT_GT_DIR,
                        help=f"Base directory containing ground truth images (organized in subfolders). Default: {DEFAULT_GT_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the SSIM results CSV. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--ssim_results_filename", type=str, default="ssim_results.csv",
                        help="Filename for the output SSIM results CSV. Default: ssim_results.csv")
    parser.add_argument("--image_size", type=str, default="256x256",
                        help="Target size (WIDTHxHEIGHT) to resize images before SSIM calculation. Default: 256x256")

    args = parser.parse_args()

    # Use resolved absolute paths for clarity
    gen_dir = os.path.abspath(args.gen_images_dir)
    gt_dir = os.path.abspath(args.gt_images_dir)
    out_dir = os.path.abspath(args.output_dir)

    print(f"Using Generated Images Dir: {gen_dir}")
    print(f"Using Ground Truth Dir:    {gt_dir}")
    print(f"Using Output Dir:          {out_dir}")

    main(
        gen_images_dir=gen_dir,
        gt_images_dir=gt_dir,
        output_dir=out_dir,
        ssim_results_filename=args.ssim_results_filename,
        image_size_str=args.image_size
    ) 