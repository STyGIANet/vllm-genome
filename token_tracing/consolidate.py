import pandas as pd
import dask.dataframe as dd
import glob
import os
import time

def consolidate_moe_data_optimized(output_filename="consolidated_moe_data.csv"):
    """
    Optimized function to merge MoE routing data, pivot layers to columns,
    and format the cell content, including Dask reading and vectorized deduplication.
    """
    print("Starting data consolidation...")
    start_time = time.time()
    
    # 1. Read and concatenate all files using Dask
    all_files = glob.glob("/tmp/vllm_moe_routing_rank_*.csv")
    if not all_files:
        print("❌ Error: No MoE routing CSV files found.")
        return

    # Use Dask for parallel reading, then compute into Pandas DataFrame
    # This is efficient for large I/O-bound tasks.
    dask_df = dd.read_csv(all_files)
    raw_df = dask_df.compute()

    print(f"✅ Successfully loaded and merged data from {len(all_files)} files. (Time: {time.time() - start_time:.2f}s)")
    
    # --- Data Cleaning and Preparation (Optimized) ---
    prep_time = time.time()

    # OPTIMIZATION 1: Vectorized creation of unique Token ID
    raw_df['Token_ID'] = (
        "Rank" + raw_df['dp_rank'].astype(str) + 
        "_Token" + raw_df['batch_token_index'].astype(str).str.zfill(6) # <-- ADD ZERO PADDING
    )
        
    # OPTIMIZATION 2: Vectorized creation of the Layer Value tuple string
    # We combine the conversion and concatenation into one step.
    raw_df['Layer_Value'] = (
        '((' + raw_df['expert_id_k0'].astype(str) + ', ' + 
        raw_df['expert_weight_k0'].astype(str) + '), ' +
        '(' + raw_df['expert_id_k1'].astype(str) + ', ' + 
        raw_df['expert_weight_k1'].astype(str) + '))'
    )
    
    print(f"✅ Data prepared with vectorized operations. (Time: {time.time() - prep_time:.2f}s)")

    # --- CRITICAL FIX: DEDUPLICATION ---
    dedup_time = time.time()
    
    # The 'pivot' failed because ('Token_ID', 'layer_name') was not unique.
    # We drop duplicates, keeping the 'last' entry found.
    # This assumes duplicate entries are either identical or the last one is desired.
    raw_df.drop_duplicates(subset=['Token_ID', 'layer_name'], keep='last', inplace=True)
    
    print(f"✅ Data deduplicated. (Time: {time.time() - dedup_time:.2f}s)")
    
    # --- Pivoting the Data ---
    pivot_time = time.time()
    # 3. Pivot the data (This should now succeed)
    pivot_df = raw_df.pivot(
        index='Token_ID',
        columns='layer_name',
        values='Layer_Value'
    )
    print(f"✅ Data pivoted. (Time: {time.time() - pivot_time:.2f}s)")
    
    # --- Final Formatting and Saving ---
    final_time = time.time()
    
    final_df = pivot_df.reset_index()

    # 4. Insert DP Rank Column (at index 1)
    final_df.insert(1, 'DP_Rank', final_df['Token_ID'].str.extract(r'Rank(\d+)').astype(int))

    final_df = final_df.rename(columns={'Token_ID': 'Composite_Token_ID'})
    
    # Sort the layers columns numerically
    layer_cols = [col for col in final_df.columns if 'model.layers' in col]
    def get_layer_number(col_name):
        try:
            return int(col_name.split('.')[2])
        except (IndexError, ValueError):
            return float('inf') 
            
    sorted_layer_cols = sorted(layer_cols, key=get_layer_number)
    final_columns = ['Composite_Token_ID', 'DP_Rank'] + sorted_layer_cols
    final_df = final_df[final_columns]
    
    # 5. Save the final file
    final_df.to_csv(output_filename, index=False)
    
    end_time = time.time()
    print(f"\n✨ Success! Consolidated and pivoted CSV saved to **{output_filename}**")
    print(f"Total Consolidation Time: {end_time - start_time:.2f}s")
    print("\n--- Final Data Structure (Head) ---")
    print(final_df[['Composite_Token_ID', 'DP_Rank', sorted_layer_cols[0]]].head(2).to_string(index=False))

if __name__ == '__main__':
    consolidate_moe_data_optimized()