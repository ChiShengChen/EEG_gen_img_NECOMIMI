import pandas as pd
import argparse
import os

def analyze_ssim(input_csv, output_per_subject_csv, output_overall_csv):
    """Reads SSIM results, calculates per-subject and overall statistics, and saves them."""
    try:
        df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_csv}")
        return
    except Exception as e:
        print(f"Error reading CSV file {input_csv}: {e}")
        return

    print(f"Read {len(df)} rows from {input_csv}")

    # 1. Statistics per subject for each model/emb_type
    try:
        # Ensure subject is treated correctly for sorting/grouping if needed
        # Extract subject number for potential numeric sorting later if desired
        # df['subject_num'] = df['subject'].str.extract(r'sub-(\d+)').astype(int)

        stats_per_subject = df.groupby(['model', 'emb_type', 'subject']).agg(
            avg_ssim=('ssim_score', 'mean'),
            std_ssim=('ssim_score', 'std'),
            count=('ssim_score', 'count') # Count of images per subject group
        ).reset_index()

        # Fill NaN in std_dev if a group has only one entry
        stats_per_subject['std_ssim'] = stats_per_subject['std_ssim'].fillna(0)

        # Optional: Sort for better readability
        stats_per_subject['subject_num'] = stats_per_subject['subject'].str.extract(r'sub-(\d+)').astype(int)
        stats_per_subject = stats_per_subject.sort_values(by=['model', 'emb_type', 'subject_num']).drop(columns=['subject_num'])

        stats_per_subject.to_csv(output_per_subject_csv, index=False, float_format='%.6f')
        print(f"Successfully saved per-subject SSIM summary to {output_per_subject_csv}")

    except KeyError as e:
        print(f"Error calculating per-subject stats: Missing column {e}. Ensure input CSV has 'model', 'emb_type', 'subject', 'ssim_score'.")
        return
    except Exception as e:
        print(f"An error occurred during per-subject aggregation or saving: {e}")
        return

    # 2. Overall statistics for each model/emb_type across all subjects
    try:
        stats_overall = df.groupby(['model', 'emb_type']).agg(
            overall_avg_ssim=('ssim_score', 'mean'),
            overall_std_ssim=('ssim_score', 'std'),
            total_image_count=('ssim_score', 'count'), # Total images for this model/emb_type
            subject_count=('subject', 'nunique') # Count of unique subjects involved
        ).reset_index()

        # Fill NaN in std_dev if a group has only one entry (less likely here)
        stats_overall['overall_std_ssim'] = stats_overall['overall_std_ssim'].fillna(0)

        # Optional: Sort for better readability
        stats_overall = stats_overall.sort_values(by=['model', 'emb_type'])

        stats_overall.to_csv(output_overall_csv, index=False, float_format='%.6f')
        print(f"Successfully saved overall SSIM summary to {output_overall_csv}")

    except KeyError as e:
        print(f"Error calculating overall stats: Missing column {e}. Ensure input CSV has 'model', 'emb_type', 'ssim_score', 'subject'.")
    except Exception as e:
        print(f"An error occurred during overall aggregation or saving: {e}")

if __name__ == "__main__":
    # Resolve defaults relative to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_INPUT = os.path.join(script_dir, 'ssim_results.csv')
    DEFAULT_OUTPUT_PER_SUBJECT = os.path.join(script_dir, 'ssim_summary_per_subject.csv')
    DEFAULT_OUTPUT_OVERALL = os.path.join(script_dir, 'ssim_summary_overall.csv')

    parser = argparse.ArgumentParser(description="Analyze SSIM results, calculating per-subject and overall statistics.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT,
                        help=f"Path to the input SSIM results CSV file. Default: {DEFAULT_INPUT}")
    parser.add_argument("--output_per_subject", type=str, default=DEFAULT_OUTPUT_PER_SUBJECT,
                        help=f"Path to save the per-subject SSIM summary CSV. Default: {DEFAULT_OUTPUT_PER_SUBJECT}")
    parser.add_argument("--output_overall", type=str, default=DEFAULT_OUTPUT_OVERALL,
                        help=f"Path to save the overall SSIM summary CSV. Default: {DEFAULT_OUTPUT_OVERALL}")

    args = parser.parse_args()

    analyze_ssim(args.input_csv, args.output_per_subject, args.output_overall) 