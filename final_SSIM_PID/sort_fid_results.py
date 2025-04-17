import pandas as pd
import argparse
import os

def sort_fid_results(input_csv, output_csv):
    """Reads, sorts, and saves FID results."""
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return
    except Exception as e:
        print(f"Error reading CSV file {input_csv}: {e}")
        return

    # Extract subject number for proper numerical sorting
    try:
        df['subject_num'] = df['subject'].str.extract(r'sub-(\d+)').astype(int)
    except Exception as e:
        print(f"Error extracting subject number from 'subject' column: {e}")
        print("Ensure the 'subject' column follows the 'sub-XX' format.")
        return

    # Define the desired order for emb_type (optional, but good for consistency)
    emb_type_order = ['eeg_embeds', 'image_embeds', 'pipe_output']
    df['emb_type'] = pd.Categorical(df['emb_type'], categories=emb_type_order, ordered=True)

    # Sort the DataFrame
    df_sorted = df.sort_values(by=['model', 'subject_num', 'emb_type'])

    # --- Calculate Averages ---
    # Group by model and embedding type, calculate mean FID and count of subjects
    averages = df_sorted.groupby(['model', 'emb_type'], observed=False).agg(
        avg_fid_score=('fid_score', 'mean'),
        std_fid_score=('fid_score', 'std'), # Calculate standard deviation
        subject_count=('subject', 'count') # Count how many subjects contribute to the average
    ).reset_index()

    # Fill NaN in std_dev if a group has only one entry (although count should reflect this)
    averages['std_fid_score'] = averages['std_fid_score'].fillna(0)

    # Prepare the averages DataFrame for concatenation
    averages_df = pd.DataFrame({
        'model': averages['model'],
        'subject': 'AVERAGE', # Identifier for average rows
        'emb_type': averages['emb_type'],
        # Put subject count in the 'num_generated_images' column for structure consistency
        'num_generated_images': averages['subject_count'],
        'fid_score': averages['avg_fid_score'],
        'std_fid_score': averages['std_fid_score'] # Add standard deviation column
    })
    # Ensure column order matches the original (excluding subject_num) + add std dev
    averages_df = averages_df[['model', 'subject', 'emb_type', 'num_generated_images', 'fid_score', 'std_fid_score']]
    # --- End Calculate Averages ---

    # Drop the temporary subject number column from the main sorted data
    df_sorted = df_sorted.drop(columns=['subject_num'])

    # Combine sorted data with the calculated averages
    # Need to ensure df_sorted also has the std_fid_score column (with NaNs perhaps) before concat if we want alignment
    # Or, just concatenate and let the std_fid_score column be NaN for non-average rows.
    # Let's choose the latter for simplicity, as std dev only makes sense for the AVERAGE rows.
    final_df = pd.concat([df_sorted, averages_df], ignore_index=True)

    # Save the final DataFrame (sorted + averages)
    try:
        # Adjust column list for saving if needed, though pandas usually handles the union of columns
        final_df.to_csv(output_csv, index=False, float_format='%.6f') # Format float for consistency
        print(f"Successfully sorted results with averages and std devs saved to {output_csv}")
    except Exception as e:
        print(f"Error writing final CSV file to {output_csv}: {e}")

if __name__ == "__main__":
    # Resolve defaults relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_INPUT = os.path.join(script_dir, 'fid_results.csv')
    DEFAULT_OUTPUT = os.path.join(script_dir, 'fid_results_sorted.csv')

    parser = argparse.ArgumentParser(description="Sort the FID results CSV file.")
    # Use the resolved absolute paths as defaults
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT,
                        help=f"Path to the input FID results CSV file. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT,
                        help=f"Path to save the sorted FID results CSV file. Default: {DEFAULT_OUTPUT}")

    args = parser.parse_args()

    # Pass the paths directly (they are either absolute defaults or user-provided)
    sort_fid_results(args.input_csv, args.output_csv) 