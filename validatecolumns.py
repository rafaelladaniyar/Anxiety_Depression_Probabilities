import pandas as pd
import numpy as np


def validate_dataframe_columns(
        df: pd.DataFrame,
        fe_instance: object,
        task_type: str,
        verbose: bool = True
    ):
    """
    This function will validate and ensure that all of the columns are safely remaining
    between the process using the instances of FeatureEngineerML and MentalHealthClassifer.

    Procedure:
        1. Will validate and obtain the list of expected columns within the DataFrame currently.
        2. Will update the feature lists to match the current DataFrame columns.
        3. Will send the information from this function to the instance object used for the class FeatureEngineeringML.

    Parameters:
        - df (pd.DataFrame): DataFrame that will be investigated and monitored for column validation throughout the entire function.
        - fe_instance (object): The instance object that was used to initialize the class FeatureEngineeringML.
        - task_type (str): choice chosen from the instance object used for initializing the class FeatureEngineeringML. Will use "_______.task_type."
        - verbose (bool, optional): Will provide additional information if requested by the user - default is "True."

    Returns:
        None

        This will import the updated list of features to the instance object for the class FeatureEngineeringML.
    """

    if verbose:
        print("Column Validation Process Beginning:")

    target_columns = {'Indicator', 'Value'}
    excluded_columns = {'Phase_Date_Specification'}

    #Provides a list of all of the columns that are currently within the DataFrame at this time.
    print(target_columns.union(excluded_columns))
    available_columns = [col for col in df.columns if col not in target_columns.union(excluded_columns)]
    available_set = set(available_columns)

    date_cols = []
    for col in available_set:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
            if verbose:
                print(f"Datetime column found: {col}")
    if date_cols:
        for col in date_cols:
            available_columns.remove(col)
    available_set = set(available_columns)
    if verbose:
        print(f"Updated Number of Columns Available: {len(available_set)}")

    if verbose:
        print(f"Number of available columns: {len(available_set)}")

    #Will obtain the original list of categorical features and numerical features that was acquired from the instance of the class "FeatureEngineeringML"
    orig_cat = getattr(fe_instance, 'categorical_features', [])
    orig_num = getattr(fe_instance, 'numerical_features', [])

    #Will now attempt to find the missing columns:
    miss_cat = [col for col in orig_cat if col not in available_set]
    miss_num = [col for col in orig_num if col not in available_set]

    if verbose and (miss_cat or miss_num):
        print(f"Missing Categorical Features Exist: {len(miss_cat)} missing")
        print(f"Missing Numerical Features Exist: {len(miss_num)} missing")

        if miss_cat:
            print("Missing Categorical Features:")
            for col in miss_cat:
                print(col)
        if miss_num:
            print("Missing Numerical Features:")
            for col in miss_num:
                print(col)

    safe_cat = []
    safe_num = []


    for col in available_columns:
        type = df[col].dtype
        unique_vals = df[col].nunique()
        total_vals = len(df[col])
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'float16', 'int16']:
            if unique_vals > 20:
                safe_num.append(col)
                continue
            elif unique_vals <= 10:
                sample_values = df[col].dropna().unique()[:5]
                if all(isinstance(x, (int, np.integer)) or (isinstance(x, float)) and x.is_integer() for x in sample_values):
                    if max(sample_values) - min(sample_values) < 20:
                        safe_cat.append(col)
                        continue
                else:
                    safe_num.append(col)
            else:
                uniqueness_ratio = unique_vals / total_vals

                if uniqueness_ratio > 0.05: 
                    safe_num.append(col)
                else:
                    safe_cat.append(col)
        elif type in ['object', 'category']:
            safe_cat.append(col)
        elif type == 'bool':
            safe_cat.append(col)
        elif 'datetime' in str(type).lower():
            safe_num.append(col)
        else:
            if unique_vals > 20:
                safe_num.append(col)
            else:
                safe_cat.append(col)
    
    orig_cat_set = set(orig_cat)
    orig_num_set = set(orig_num)

    conflicts_resolved = 0
    for col in available_columns:
        if col in orig_cat_set and col in safe_num:
            ## What it means that if it was in the categorical feature but we had labeled it as numerical right now.

            if df[col].nunique() <=15:
                safe_num.remove(col)
                safe_cat.append(col)
                conflicts_resolved += 1
                if verbose:
                    print(f"Conflict was resolved for column: {col}")
        elif col in orig_num_set and col in safe_cat:
            # Originally located in numerical features but currently labeled it as a categorical feature.
            if df[col].dtype in ['int64', 'int32', 'float64', 'float32']:
                safe_cat.remove(col)
                safe_num.append(col)
                conflicts_resolved += 1
                if verbose:
                    print(f"Conflict was resolved for column: {col}")
    if verbose and conflicts_resolved > 0:
        print(f"Total Number of Conflicts resolved: {conflicts_resolved}")

    fe_instance.categorical_features = safe_cat.copy()
    fe_instance.numerical_features = safe_num.copy()

