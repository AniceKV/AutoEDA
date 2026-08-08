import os
import json
import re
import pandas as pd
import numpy as np
import hashlib
from typing import Optional, List, Dict, Any, Tuple


class DataProfiler:
    """
    Classful Data Profiler for algorithmic dataset inspection, statistical profiling,
    distribution classification, and filesystem profile caching.
    """

    def __init__(self, cache_dir: str = ".cache/autoeda"):
        self.cache_dir = cache_dir

    def is_non_distributional_column(self, col_name: str, series: Optional[pd.Series] = None) -> bool:
        """
        Determines if a column lacks meaningful univariate distribution properties.
        Excludes unique IDs, spatial coordinates (latitude, longitude), timestamps/dates,
        sampling weights (e.g. fnlwgt), and high-cardinality non-informative key columns (e.g. Ticket).
        """
        col_clean = str(col_name).strip().lower()

        # 1. Spatial Coordinate column patterns
        coord_patterns = [
            r"^(latitude|longitude|lat|lng|lon)$",
            r".*_(latitude|longitude|lat|lng|lon)$",
            r"^(latitude|longitude|lat|lng|lon)_.*",
            r".*(coord|coordinates|location_lat|location_lng|geo_lat|geo_lon).*"
        ]
        for pattern in coord_patterns:
            if re.match(pattern, col_clean):
                return True

        # 2. Identifier / Key / Sampling Weight column name patterns
        id_patterns = [
            r"^(id|uuid|guid|pk|index|row_id|row_num|code|hash|ssn|fnlwgt|ticket|ticket_num|ticket_no|passengerid|name|member_id)$",
            r".*(_id|_uuid|_guid|_pk|_code|_hash)$",
            r"^(id_|uuid_|guid_|pk_|ticket_).*"
        ]
        matches_id_pattern = False
        for pattern in id_patterns:
            if re.match(pattern, col_clean):
                matches_id_pattern = True
                break
                
        if matches_id_pattern:
            if series is None:
                # Conservative fallback: respect the name match if no data is provided
                return True
            else:
                s = series.dropna()
                if len(s) == 0:
                    return True
                
                # A true ID shouldn't be a float (scientific index scores often are)
                if pd.api.types.is_float_dtype(s):
                    pass # Continue to generic data heuristics instead of dropping
                
                # A true ID should have very high cardinality. If it's low, it's likely categorical
                elif len(s) > 20 and (s.nunique() / len(s)) < 0.5:
                    pass # Continue to generic data heuristics instead of dropping
                
                else:
                    return True

        # 3. Temporal / Timestamp column patterns
        time_patterns = [
            r"^(date|time|timestamp|created_at|updated_at|datetime)$",
            r".*(_date|_time|_timestamp|_at)$"
        ]
        for pattern in time_patterns:
            if re.match(pattern, col_clean):
                return True

        # 4. Data Series Heuristics (if series data provided)
        if series is not None:
            s = series.dropna()
            if len(s) == 0:
                return True

            # Check datetime dtype
            if pd.api.types.is_datetime64_any_dtype(s):
                return True

            # Strictly sequential integer index check (e.g. 1, 2, 3, 4 ... N)
            if pd.api.types.is_numeric_dtype(s) and len(s) > 10:
                diffs = s.diff().dropna()
                if len(diffs) > 0 and (diffs == 1).all():
                    return True

            # High cardinality non-numeric / string check
            if len(s) > 20:
                uniq_ratio = s.nunique() / len(s)
                if not pd.api.types.is_numeric_dtype(s):
                    if uniq_ratio > 0.30 or s.nunique() > 50:
                        return True
                elif uniq_ratio > 0.90 and pd.api.types.is_integer_dtype(s):
                    # Almost completely unique integer column
                    if any(kw in col_clean for kw in ["id", "wt", "weight", "code", "fnlwgt", "num", "no"]):
                        return True

        return False

    def calculate_column_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
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
            unique_pct = (cardinality / num_rows) * 100 if num_rows > 0 else 0.0

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
                    col_kurt = df[col].kurtosis()

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
                col_min = col_max = col_mean = col_std = col_median = col_q1 = col_q3 = col_iqr = col_skew = col_kurt = None
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
                "skew": float(col_skew) if pd.notnull(col_skew) else None,
                "kurtosis": float(col_kurt) if pd.notnull(col_kurt) else None,
                "unique_pct": round(unique_pct, 2)
            })

        return stats_list

    def profile_dataframe(self, df: pd.DataFrame, data_path: str = "") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Profiles an in-memory DataFrame and returns (llm_context_summary, full_profile).
        """
        num_rows, num_cols = df.shape
        stats_list = self.calculate_column_stats(df)

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

        full_profile = {
            "dataset_name": os.path.basename(data_path) if data_path else "dataframe",
            "dimensions": {"rows": num_rows, "columns": num_cols},
            "missing_values_summary": llm_context_summary["missing_values_summary"],
            "schema": llm_context_summary["schema"],
            "columns": [
                {
                    "column": col["column"],
                    "dtype": col["dtype"],
                    "missing_count": col["missing_count"],
                    "missing_pct": col["missing_pct"],
                    "cardinality": col["cardinality"],
                    "unique_pct": col["unique_pct"],
                    "mean": col["mean"],
                    "median": col["median"],
                    "std": col["std"],
                    "min": col["min"],
                    "max": col["max"],
                    "q1": col["q1"],
                    "q3": col["q3"],
                    "iqr": col["iqr"],
                    "skew": col["skew"],
                    "kurtosis": col["kurtosis"]
                }
                for col in stats_list
            ]
        }

        return llm_context_summary, full_profile

    def run_and_save_profile(self, data_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Algorithmically profiles the dataset, saves the complete structured 
        statistics to JSON in the output directory, and returns the LLM metadata context.
        Includes caching based on dataset path, size, and modification time.
        """
        cache_path = None
        try:
            file_stat = os.stat(data_path)
            hash_key = hashlib.md5(f"{data_path}_{file_stat.st_size}_{file_stat.st_mtime}".encode()).hexdigest()
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_path = os.path.join(self.cache_dir, f"profile_{hash_key}.json")

            if os.path.exists(cache_path):
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)

                os.makedirs(output_dir, exist_ok=True)
                profile_save_path = os.path.join(output_dir, "metadata_profile.json")
                with open(profile_save_path, "w", encoding="utf-8") as f:
                    json.dump(cached_data["full_profile"], f, indent=4)
                print(f"Loaded cached profile from {cache_path}")
                return cached_data["llm_context_summary"]
        except Exception:
            cache_path = None

        try:
            df = pd.read_csv(data_path)
        except Exception as e:
            return {"error": f"Error loading dataset: {str(e)}"}

        llm_context_summary, full_profile = self.profile_dataframe(df, data_path=data_path)

        os.makedirs(output_dir, exist_ok=True)
        profile_save_path = os.path.join(output_dir, "metadata_profile.json")
        with open(profile_save_path, "w", encoding="utf-8") as f:
            json.dump(full_profile, f, indent=4)

        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "llm_context_summary": llm_context_summary,
                        "full_profile": full_profile
                    }, f)
            except Exception:
                pass

        print(f"Programmatic profile successfully written: {profile_save_path}")

        return llm_context_summary


# Default Singleton Instance for Backward Compatibility
default_profiler = DataProfiler()


def is_non_distributional_column(col_name: str, series: Optional[pd.Series] = None) -> bool:
    return default_profiler.is_non_distributional_column(col_name, series)


def calculate_column_stats(df: pd.DataFrame) -> list:
    return default_profiler.calculate_column_stats(df)


def run_and_save_profile(data_path: str, output_dir: str) -> dict:
    return default_profiler.run_and_save_profile(data_path, output_dir)


if __name__ == "__main__":
    print("Classful DataProfiler v3 initialized.")
