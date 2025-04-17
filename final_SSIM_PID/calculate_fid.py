import os
import glob
import re
import subprocess
import shutil
import argparse
import pandas as pd

# Regex to parse generated image filenames (same as in the original script)
FILENAME_REGEX = re.compile(
    r"^(.*?)_sub-?(\d{2})_generated_image_(\d+)_from_(eeg_embeds|image_embeds|pipe_output)(?:_\S+)?\.png$"
)

def run_fid(path1, path2, device='cuda', batch_size=50, num_workers=0):
    """Runs the pytorch-fid command and returns the FID score."""
    command = [
        'python', '-m', 'pytorch_fid',
        path1, path2,
        '--device', device,
        '--batch-size', str(batch_size),
        '--num-workers', str(num_workers)
    ]
    print(f"Running FID command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        print(f"FID Output:\n{output}")
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

def prepare_flat_gt_dir(gt_base_dir, temp_base_dir):
    """Creates a temporary directory with all GT images flattened."""
    temp_gt_flat_dir = os.path.join(temp_base_dir, 'temp_gt_flat')
    os.makedirs(temp_gt_flat_dir, exist_ok=True)
    print(f"\nPreparing flattened ground truth directory: {temp_gt_flat_dir}")

    all_gt_files = glob.glob(os.path.join(gt_base_dir, '**', '*.jpg'), recursive=True)
    all_gt_files.extend(glob.glob(os.path.join(gt_base_dir, '**', '*.png'), recursive=True))

    if not all_gt_files:
         print(f"Error: No ground truth image files found in {gt_base_dir} or its subdirectories.")
         return None

    gt_copied_count = 0
    for gt_src_path in all_gt_files:
        try:
            dir_name = os.path.basename(os.path.dirname(gt_src_path))
            base_name = os.path.basename(gt_src_path)
            sanitized_dir_name = re.sub(r'[\\/*?:"<>|]', "_", dir_name)
            gt_dst_path = os.path.join(temp_gt_flat_dir, f"{sanitized_dir_name}_{base_name}")

            # Avoid re-copying if not necessary, though shutil.copy2 handles this
            # with open(gt_src_path, 'rb') as fsrc, open(gt_dst_path, 'wb') as fdst:
            #     fdst.write(fsrc.read())
            shutil.copy2(gt_src_path, gt_dst_path) # copy2 preserves metadata if possible
            gt_copied_count += 1
        except Exception as e:
            print(f"Error copying ground truth file {gt_src_path} to {temp_gt_flat_dir}: {e}")

    print(f"Successfully copied {gt_copied_count} ground truth images to {temp_gt_flat_dir}")
    if gt_copied_count == 0:
        print("Error: Failed to copy any ground truth images for FID.")
        return None

    return temp_gt_flat_dir


def main_filtered(filtered_image_groups, gt_images_dir, output_dir, fid_results_filename, device, batch_size, num_workers):
    """ Modified main function to accept pre-filtered image groups """
    os.makedirs(output_dir, exist_ok=True)
    temp_fid_dir_base = os.path.join(output_dir, 'temp_fid_dirs_nerv2_perturbed') # Adjusted temp dir name
    os.makedirs(temp_fid_dir_base, exist_ok=True)

    # Prepare flattened GT directory ONCE
    gt_fid_path_flat = prepare_flat_gt_dir(gt_images_dir, temp_fid_dir_base)
    if not gt_fid_path_flat:
        print("Aborting FID calculation due to error in preparing GT directory.")
        try:
            shutil.rmtree(temp_fid_dir_base)
        except Exception: pass
        return

    # Calculate FID for each filtered group
    fid_data = []
    print(f"\nCalculating FID scores for filtered groups (comparing each group to flattened GT: {gt_fid_path_flat})...")

    for group_key, gen_paths in filtered_image_groups.items(): # Use the filtered groups passed in
        model, subject, emb_type = group_key
        print(f"\nProcessing FID for: {model}, {subject}, {emb_type} ({len(gen_paths)} images)")

        # Create a unique temporary directory for this specific group
        group_fid_dir = os.path.join(temp_fid_dir_base, f"{model}_{subject}_{emb_type}")
        os.makedirs(group_fid_dir, exist_ok=True)

        print(f"Copying {len(gen_paths)} generated images to temporary directory: {group_fid_dir}")
        copied_count = 0
        for src_path in gen_paths:
             try:
                 dst_path = os.path.join(group_fid_dir, os.path.basename(src_path))
                 shutil.copy2(src_path, dst_path)
                 copied_count += 1
             except Exception as e:
                 print(f"Error copying {src_path} to {group_fid_dir}: {e}")

        if copied_count == 0:
            print(f"Warning: No images copied for group {group_key}. Skipping FID calculation.")
            try:
                os.rmdir(group_fid_dir)
            except OSError as e:
                print(f"Could not remove empty temp dir {group_fid_dir}: {e}")
            continue

        print(f"Successfully copied {copied_count} generated images.")

        # Calculate FID score using the FLATTENED GT path
        fid_score = run_fid(group_fid_dir, gt_fid_path_flat, device, batch_size, num_workers)

        if fid_score is not None:
            fid_data.append({
                'model': model,
                'subject': subject,
                'emb_type': emb_type,
                'num_generated_images': len(gen_paths),
                'fid_score': fid_score
            })
            print(f"FID score for {group_key}: {fid_score:.4f}")
        else:
             print(f"FID calculation failed for group {group_key}.")

    # Save FID Results
    if fid_data:
        fid_df = pd.DataFrame(fid_data)
        output_path = os.path.join(output_dir, fid_results_filename)
        fid_df.to_csv(output_path, index=False)
        print(f"\nFID results saved to {output_path}")
    else:
        print("\nNo FID scores were calculated successfully.")

    # Clean up ALL temporary directories used for FID
    print(f"\nCleaning up base temporary FID directory: {temp_fid_dir_base}")
    try:
        shutil.rmtree(temp_fid_dir_base)
        print("Successfully cleaned up temporary FID directories.")
    except Exception as e:
        print(f"Error cleaning up base temporary FID directory {temp_fid_dir_base}: {e}")

    print("\nFID calculation script finished.")


if __name__ == "__main__":
    # Define default paths based on previous script structure
    DEFAULT_GEN_DIR = '/home/aidan/EEG_Image_decode_old/Generation/gen_images_perturbed_visual' # Updated path
    DEFAULT_GT_DIR = '/home/aidan/EEG_Image_decode_old/image_set/test_images'
    DEFAULT_OUTPUT_DIR = '/home/aidan/EEG_Image_decode_old/final_SSIM_PID' # Or perhaps '.' if you prefer current dir
    DEFAULT_FID_RESULTS_FILENAME = 'fid_results_nerv2_perturbed_visual.csv' # Updated filename

    parser = argparse.ArgumentParser(description="Calculate FID scores between generated images and ground truth images.")
    parser.add_argument("--gen_images_dir", type=str, default=DEFAULT_GEN_DIR,
                        help=f"Directory containing the generated images. Default: {DEFAULT_GEN_DIR}")
    parser.add_argument("--gt_images_dir", type=str, default=DEFAULT_GT_DIR,
                        help=f"Base directory containing the ground truth images (organized in subfolders). Default: {DEFAULT_GT_DIR}")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the results and temporary files. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--fid_results_filename", type=str, default=DEFAULT_FID_RESULTS_FILENAME, help="Filename for the output FID results CSV.") # Use updated default
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for FID calculation ('cuda' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for FID calculation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for FID calculation.")

    args = parser.parse_args()

    # Filter image groups before processing
    all_gen_files_main = glob.glob(os.path.join(args.gen_images_dir, '**', "*.png"), recursive=True)
    print(f"Found {len(all_gen_files_main)} generated images in {args.gen_images_dir} (including subdirectories)")

    if not all_gen_files_main:
        print("No generated files found. Exiting.")
        exit() # Exit if no files

    image_groups_main = {} # To group images for FID: {(model, subject, emb_type): [paths]}
    for gen_path in all_gen_files_main:
        filename = os.path.basename(gen_path)
        match = FILENAME_REGEX.match(filename)

        if not match:
            if "generated_image" in filename:
                 print(f"Warning: Could not parse generated filename: {filename}")
            continue

        model, subject_num_str, _, emb_type = match.groups()
        subject_num = int(subject_num_str) # Convert to int for comparison
        subject = f"sub-{subject_num_str}" # Keep original string format

        # --- Filtering ---
        if model == 'Nerv2' and 1 <= subject_num <= 10:
            group_key = (model, subject, emb_type)
            if group_key not in image_groups_main:
                image_groups_main[group_key] = []
            image_groups_main[group_key].append(gen_path)
        # --- End Filtering ---

    if not image_groups_main:
        print("No image groups found matching the criteria (nerv2, sub-01 to sub-10). Exiting.")
        exit()

    print(f"Filtered down to {len(image_groups_main)} groups for FID calculation.")

    # Use resolved absolute paths for clarity and robustness
    gen_dir = os.path.abspath(args.gen_images_dir) # Keep gen_dir as the original input for prepare_flat_gt_dir if needed later, though it's not used by it
    gt_dir = os.path.abspath(args.gt_images_dir)
    out_dir = os.path.abspath(args.output_dir)

    print(f"Using Generated Images Dir: {gen_dir}") # Info purpose
    print(f"Using Ground Truth Dir:    {gt_dir}")
    print(f"Using Output Dir:          {out_dir}")

    # Call main with the filtered groups
    main_filtered(
        filtered_image_groups=image_groups_main, # Pass the filtered groups
        gt_images_dir=gt_dir,
        output_dir=out_dir,
        fid_results_filename=args.fid_results_filename,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    ) 