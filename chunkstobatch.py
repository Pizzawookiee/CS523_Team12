import os
import pandas as pd
from fastparquet import write

def process_parquet_files_in_batches(output_directory, batch_size=5):
    # Get all the parquet files in the output directory
    all_files = [file for file in os.listdir(output_directory) if file.endswith('.parquet')]
    
    # Iterate over files in batches
    for i in range(0, len(all_files), batch_size):
        batch_files = all_files[i:i+batch_size]  # Select the current batch of files
        batch_dfs = []  # List to hold DataFrames of current batch

        # Read each file in the batch
        for batch_file in batch_files:
            batch_df = pd.read_parquet(os.path.join(output_directory, batch_file))  # Read each parquet file
            batch_dfs.append(batch_df)

        # Concatenate batch DataFrames into one
        batch_df_combined = pd.concat(batch_dfs, ignore_index=True)

        # Here you can process the batch DataFrame (e.g., resample, clean, analyze, etc.)
        print(f"Processed batch {i // batch_size + 1}")

        # Save the batch as a final output if needed (optional)
        final_output_path = f"{output_directory}/batch_{i // batch_size + 1}_final.parquet"
        batch_df_combined.to_parquet(final_output_path, compression='brotli', index=False)

        # Optionally: Clear memory after each batch
        del batch_df_combined
        del batch_dfs
        import gc
        gc.collect()

# Example usage
output_directory = r'C:\Users\damol\Downloads\Competition\METAR_parquet_chunks'  # Change to your directory
process_parquet_files_in_batches(output_directory)
