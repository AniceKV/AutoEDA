import os
import json
import pandas as pd
import numpy as np

def calculate_column_stats(df: pd.DataFrame) -> list:
    """
    Computes profiling stats for each column in the DataFrame.
    Returns a list of dictionaries with metadata and properties.
    """
    num_rows = len(df)
    stats_list = []
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        missing_count = int(df[col].isnull().sum())
        missing_pct = (missing_count / num_rows) * 100 if num_rows > 0 else 0.0
        cardinality = df[col].nunique()
        
        properties = []
        # Exclude boolean types as pandas considers bool numeric, but numpy quantile fails on boolean subtract
        is_num = pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_bool_dtype(df[col])
        if is_num:
            try:
                # Numeric column metrics
                col_min = df[col].min()
                col_max = df[col].max()
                col_mean = df[col].mean()
                col_std = df[col].std()
                col_median = df[col].median()
                col_q1 = df[col].quantile(0.25)
                col_q3 = df[col].quantile(0.75)
                col_iqr = col_q3 - col_q1 if pd.notnull(col_q3) and pd.notnull(col_q1) else None
                col_skew = df[col].skew()
                
                # Format outputs gracefully checking for null values
                min_str = f"{col_min:.2f}" if pd.notnull(col_min) and isinstance(col_min, (int, float, np.number)) else str(col_min)
                max_str = f"{col_max:.2f}" if pd.notnull(col_max) and isinstance(col_max, (int, float, np.number)) else str(col_max)
                mean_str = f"{col_mean:.2f}" if pd.notnull(col_mean) and isinstance(col_mean, (int, float, np.number)) else "N/A"
                median_str = f"{col_median:.2f}" if pd.notnull(col_median) and isinstance(col_median, (int, float, np.number)) else "N/A"
                
                properties.append(f"Range: [{min_str} to {max_str}]")
                properties.append(f"Mean: {mean_str}")
                properties.append(f"Median: {median_str}")
                if pd.notnull(col_skew) and abs(col_skew) > 1:
                    properties.append(f"Highly Skewed ({col_skew:.2f})")
            except Exception:
                is_num = False

        if not is_num:
            col_min = col_max = col_mean = col_std = col_median = col_q1 = col_q3 = col_iqr = col_skew = None
            # Categorical/Object/Boolean column metrics
            top_values = df[col].value_counts().head(3).to_dict()
            val_strs = [f"'{k}': {v}" for k, v in top_values.items()]
            properties.append(f"Top Values: {', '.join(val_strs)}")
            
        missing_str = f"{missing_count} ({missing_pct:.1f}%)" if missing_count > 0 else "0 (0.0%)"
        
        stats_list.append({
            "column": col,
            "dtype": dtype,
            "missing": missing_str,
            "cardinality": cardinality,
            "properties": " | ".join(properties),
            "missing_count": missing_count,
            "missing_pct": round(missing_pct, 2),
            "mean": float(col_mean) if pd.notnull(col_mean) else None,
            "std": float(col_std) if pd.notnull(col_std) else None,
            "median": float(col_median) if pd.notnull(col_median) else None,
            "min": float(col_min) if pd.notnull(col_min) else None,
            "max": float(col_max) if pd.notnull(col_max) else None,
            "q1": float(col_q1) if pd.notnull(col_q1) else None,
            "q3": float(col_q3) if pd.notnull(col_q3) else None,
            "iqr": float(col_iqr) if pd.notnull(col_iqr) else None,
            "skew": float(col_skew) if pd.notnull(col_skew) else None
        })
        
    return stats_list

def run_and_save_profile(data_path: str, output_dir: str) -> dict:
    """
    Algorithmically profiles the dataset, saves the complete structured 
    statistics (matching the LLM expectations) to JSON in the sandbox directory, 
    and returns the simplified metadata dict needed to construct the LLM prompt.
    """
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        return {"error": f"Error loading dataset: {str(e)}"}
        
    num_rows, num_cols = df.shape
    stats_list = calculate_column_stats(df)
    
    # 1. Build the high-level summary that matches exactly what the LLM expects
    llm_context_summary = {
        "dimensions": f"{num_rows} rows x {num_cols} columns",
        "missing_values_summary": {
            col["column"]: col["missing"]
            for col in stats_list if col["missing_count"] > 0
        },
        "schema": {
            col["column"]: {
                "type": col["dtype"],
                "cardinality": col["cardinality"],
                "key_metric": col["properties"]
            } for col in stats_list
        }
    }
    
    # 2. Add raw data metadata so both structures are stored in metadata_profile.json
    full_profile = {
        "dataset_name": os.path.basename(data_path),
        "dimensions": {"rows": num_rows, "columns": num_cols},
        "missing_values_summary": llm_context_summary["missing_values_summary"],
        "schema": llm_context_summary["schema"],
        "columns": [
            {
                "column": col["column"],
                "dtype": col["dtype"],
                "missing_count": col["missing_count"],
                "missing_pct": col["missing_pct"],
                "cardinality": col["cardinality"]
            }
            for col in stats_list
        ]
    }
    
    # 3. Save the full unified profile directly inside the sandbox directory
    os.makedirs(output_dir, exist_ok=True)
    profile_save_path = os.path.join(output_dir, "metadata_profile.json")
    with open(profile_save_path, "w", encoding="utf-8") as f:
        json.dump(full_profile, f, indent=4)
        
    print(f"Programmatic profile successfully written: {profile_save_path}")
    
    return llm_context_summary

if __name__ == "__main__":
    print("Dataset Profiler v3 initialized.")
