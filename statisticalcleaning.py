import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class StatisticalCleaning:
    """
    This class will implement basic statistical cleaning methods and imputation methods into the necessary DataFrame
    (in this case, the mental health dataset). Will be focusing on implementing and making decisions for feature retention,
    and generating imputations where necessary. Target variable has been already initially cleaned to removed all null values,
    but outliers may still remain, and null values in other columns still remain as well.

    Here is an outline of all of the methods that are included within the class:
        - self.revealing_possible_null_rates(df, drop_col=['Phase_Date_Specification'])
        - self._analyze_quartile_range_pattern(df)
        - self.outlier_detection_on_target(df, target_variable='Value', drop_col=['Phase_Date_Specification'])
        - self._analyze_outlier_values(df, outliers, target_variable='Value', drop_col=['Phase_Date_Specification'])
        - self.apply_quartile_imputation_method(df, group_cols=['Group', 'Subgroup', 'Phase', 'Time_Period'], target_variable='Value', drop_col=['Phase_Date_Specification'], use_log=False)
    """
    def __init__(self):
        self.missing_threshold = 0.3 # General rule-of-thumb implemented for data analysts
        self.outlier_bounds = {}
        self.outlier_indices_lists = {}
    def revealing_possible_null_rates(
            self,
            df: pd.DataFrame,
            drop_col: list[str] = ['Phase_Date_Specification']
            ):
        """
        Will present information on all of the null rates that currently remain at this time for all of the columns, if they exist
        at this time. If there are no null values in the DataFrame, will return statement that DataFrame has no null values at this time.

        Parameters:
            - df (pd.DataFrame): DataFrame that is currently being investigated
            - drop_col (list[str], optional): Program will read through this list to drop certain columns from the DataFrame for statistical calculations.

        Returns:
            - None \n
            If there are null values in the DataFrame, will return columns with their respective null rates.
            If there are no null values in the DataFrame, will return statement that there are no null values in the DataFrame.
        """
        modified_df = df.copy()
        #Will retain the original copy of the columns to be implemented back within the dataset after the changes were made or calculations were completed.
        saved_columns = modified_df[drop_col]
        #Fail-Safe within the program - even if the user attempts to remove the column from the parameter, the column 'Phase_Date_Specification' will be dropped.
        if 'Phase_Date_Specification' not in drop_col:
            drop_col = drop_col + ['Phase_Date_Specification']
            saved_column = modified_df[drop_col]

        #Will drop all of the columns that are specified by the user/original parameter if parameter was not modified. This will ensure that those columns are not 
        #considered within the statistical calculations as they will not be implemented within the ML model itself as a feature.
        if len(drop_col) > 0:
            modified_df = modified_df.drop(columns=drop_col, errors='ignore')

        missing_values = {}
        for column in modified_df.columns:
            total_missing = modified_df[column].isnull().sum()
            if total_missing != 0:
                null_rate = (total_missing / modified_df[column].shape[0]) * 100
                missing_values[column] = {
                    'Count': total_missing,
                    'Missing_Percentage': null_rate
                }

                del null_rate, column, total_missing, modified_df, df

        if len(missing_values) > 0:
            print("Current Null Rates in DataFrame at this Time:")
            for col, stats in missing_values.items():
                print(f"Column '{col:10s}': {stats['Count']:5d} ({stats['Missing_Percentage']:5.1f}%)")

            del col, stats

        else:
            print("There are no null values in the DataFrame at this time.")


        return missing_values
    def _analyze_quartile_range_pattern(
            self,
            df: pd.DataFrame
            ):
        """
        Specifically analyzes the column "Quartile_Range" as we are aware that this is the specific column that has been
        the only column that is giving us issues with null values at this time. It will provide us a statistical analysis
        of the column.

        Parameters:
            - df: (pd.DataFrame) DataFrame that will be analyze statistically to be able to help with determination if imputation
            can be implemented on the column 'Quartile_Range'. Will be verified to ensured that the column 'Quartile_Range' does exist
            within the program.
        """
        modified_df = df.copy()

        assert 'Quartile_Range' in modified_df.columns, AssertionError; "DataFrame currently does not contain the contain 'Quartile_Range'. Function cannot be implemented at this time." 
        qr_missing_per_group = modified_df.groupby('Group').apply(
            lambda x: (x['Quartile_Range'].isnull() | (x['Quartile_Range'] == '')).sum()
        )
        qr_total_per_group = modified_df.groupby('Group').size()
        qr_percentage_per_group = ((qr_missing_per_group / qr_total_per_group) * 100).round(1)

        print("Reviewing Groups Missing Data for Column 'Quartile_Range':")
        print(f"Number of Unique Groups in Dataset: {len(modified_df['Group'].unique())}")
        missing_count = 0
        for group in modified_df['Group'].unique():
            missing = qr_missing_per_group[group]
            total = qr_total_per_group[group]
            pct = qr_percentage_per_group[group]
            if pct > 0.00:
                missing_count += 1
                print(f"{group:35s}: {missing:4d}/{total:4d} ({pct:5.2f}%)")
        print(f"Number of Unique Groups with Missing 'Quartile_Range' Data: {missing_count}")
        print()
        qr_data = modified_df['Quartile_Range'].notna() & (modified_df['Quartile_Range'] != '')
        existing = modified_df[qr_data]
        print(f"Number of Rows with Data for Column 'Quartile_Range': {len(existing)}")
        print("Groups DataFrame with data for 'Quartile_Range' (with associated percentages): ")
        for group in existing['Group'].unique():
            print(group)
        print()
        if len(existing) > 0:
            print("Statistics for Information with Data in 'Quartile_Range' in Correlation to other Columns: ")
            print(f"Target Variable ('Value') mean: {existing['Value'].mean():.3f}")
            print(f"Target Variable ('Value') standard deviation: {existing['Value'].std():.3f}")
            print(f"Target Variable ('Value') median: {existing['Value'].median():.3f}")
            if ('Low_CI' in existing.columns) and ('High_CI' in existing.columns):
                print(f"'Low_CI' Example: {existing['Low_CI'].iloc[0]}")
                print(f"'High_CI' Example: {existing['High_CI'].iloc[0]}")
            
            no_qr_but_yes_ci = (~qr_data) & (modified_df['Low_CI'].notna() & modified_df['High_CI'].notna())
            recoverable = no_qr_but_yes_ci.sum()
            if recoverable > 0:
                recover_rate = (recoverable / (~qr_data).sum()) * 100
                print(f"There is a potential recovery rate of {recover_rate:.2f}% for the 'Quartile_Range' column.")
        else:
            print("There is no existing data that could be found for the column 'Quartile_Range' column.")

        return None
    def outlier_detection_on_target(
            self,
            df: pd.DataFrame,
            target_variable: str = 'Value',
            drop_col: list[str] = ['Phase_Date_Specification']
            ):
        """
        Conducts statistical analysis as well as detects all potential outliers within the specified target column. Used only
        for the target column that is meant for the regression-based ML model. Will find outliers through the Interquartile Range (IQR)
        method, Z-score method,  and percentile method. Will also depict a visualization of a boxplot to help see if there are potential outliers.

        Parameters:
            - df (pd.DataFrame): DataFrame that will be analyzed at the time and where the target variable is located.
            - target_variable (string, optional): Name of the target column. Will be analyzed and searched for outliers. If outliers are found, will be
            investigated further to determine if to remove from the DataFrame.
            - drop_col (list[str], optional): Program will read through this list to drop certain columns from the dataframe for statistical calculations.
            If the function is returning a DataFrame, the columns will be reinserted back.
        
        Returns:
            - self.outlier_indices_lists (dict{str:list[int]}): A dictionary which contains the for the key the name of the method that was used to find the outliers and the value the indices
            of the outliers which were found through that specific method.The list of integers can later be used to analyze the characteristics of those specific entries or to
            remove those indices completely from the DataFrame structure.
        """
        modified_df = df.copy()
        #Will retain the original copy of the columns to be implemented back within the dataset after the changes were made or calculations were completed.
        saved_columns = modified_df[drop_col]
        #Fail-Safe within the program - even if the user attempts to remove the column from the parameter, the column 'Phase_Date_Specification' will be dropped.
        if 'Phase_Date_Specification' not in drop_col:
            drop_col = drop_col + ['Phase_Date_Specification']
            saved_column = modified_df[drop_col]

        #Will drop all of the columns that are specified by the user/original parameter if parameter was not modified. This will ensure that those columns are not 
        #considered within the statistical calculations as they will not be implemented within the ML model itself as a feature.
        if len(drop_col) > 0:
            modified_df = modified_df.drop(columns=drop_col, errors='ignore')

        assert target_variable in modified_df.columns, AssertionError; "Please verify that 'target_variable' is a column name from the DataFrame."

        null_rate = round((modified_df[target_variable].isnull().sum() / modified_df[target_variable].shape[0]) * 100, 3)
        np.testing.assert_allclose(null_rate, 0.000, atol=1e-03), "Please verify that 'target_variable' column has been cleaned prior to inserting into function."
        del null_rate
        
        print(f"Overview of Statistical Analysis for Column '{target_variable}':")
        print()
        print(f"Mean of Column Currently: {modified_df[target_variable].mean():.3f}")
        print(f"Standard Deviation of Column Currently: {modified_df[target_variable].std():.3f}")
        print(f"Median of Column Currently: {modified_df[target_variable].median():.3f}")
        print(f"Minimum Value found in Column Currently: {modified_df[target_variable].min()}")
        print(f"Maximum Value found in Column Currently: {modified_df[target_variable].max()}")

        pearson_second_coeff = (3 * (modified_df[target_variable].mean() - modified_df[target_variable].median())) / (modified_df[target_variable].std())
        print(f"Pearson's Second Coefficient (Testing for Skewness): {pearson_second_coeff:.3f}")
        if (pearson_second_coeff > -0.5) & (pearson_second_coeff < 0.5):
            print("Interpretation of Skewness Currently: Approximately Symmetric")
        elif (pearson_second_coeff < -0.5):
            print(f"Interpretation of Skewness for '{target_variable}' Column: NEGATIVE SKEWNESS")
            print("Results: Left tail of distribution is currently longer than right tail of distribution.")
        elif (pearson_second_coeff > 0.5):
            print(f"Interpretation of Skewness for '{target_variable}' Column: POSITIVE SKEWNESS")
            print("Results: Right tail of distribution is currently longer than left tail of distribution.")
        if (pearson_second_coeff < -1.0) or (pearson_second_coeff > 1.0):
            print("Magnitude of Skewness is found to be very large at this time. Dataset is found to be very skewed.")
        
        target = modified_df[target_variable]
        plt.figure(figsize=(12,8))
        sns.boxplot(x=target, fliersize=10, flierprops={"marker": "x", "markerfacecolor": "red", "markeredgecolor": "black"})
        plt.title(f"Boxplot of '{target_variable}' Column", fontsize=20, fontweight='bold', fontstyle='oblique')
        plt.xlabel('Value')
        plt.show()
        print()

        print("Method 1 - IQR Calculations:")
        ## Will Find 
        quantile1 = target.quantile(q=0.25)
        print("Lower Quantile: {0:.3f}".format(quantile1))

        quantile3 = target.quantile(q=0.75)
        print("Upper Quantile: {0:.3f}".format(quantile3))

        iqr = quantile3 - quantile1
        print("IQR Found: {0:.3f}".format(iqr))

        lower_bound = quantile1 - (1.5 * iqr)
        upper_bound = quantile3 + (1.5 * iqr)
        print(f"Lower bound limit for values in '{target_variable}' column: {lower_bound:.3f}")
        print(f"Upper bound limit for values in '{target_variable}' column: {upper_bound:.3f}")
        self.outlier_bounds = {
            'IQR Lower Bound' : lower_bound,
            'IQR Upper Bound' : upper_bound
        }
        iqr_outlier_count = 0
        iqr_outlier_index_list = []
        for i in range(target.shape[0]):
            if target.iloc[i] < lower_bound:
                iqr_outlier_count += 1
                iqr_outlier_index_list.append(i)
            elif target.iloc[i] > upper_bound:
                iqr_outlier_count += 1
                iqr_outlier_index_list.append(i)
            else:
                continue
        print(f"Number of Outliers found in '{target_variable}' Column: {iqr_outlier_count} ({(iqr_outlier_count/target.shape[0])*100:.3f} %)")
        
        print()

        print("Method 2 - Z Score Calculations:")
        ## Calculated by finding z = (X - mu) / sigma 
        ## X = Value of the Element
        ## Mu = Population Mean
        ## Sigma = Population Standard Deviation
        ## z = z-scores (what we are trying to find at this time)
        z_scores = np.abs((target - target.mean()) / target.std())
        z_threshold = 3
        z_outlier_count = 0
        counter = 0
        z_outlier_index_list = []
        for score in z_scores:
            if score > z_threshold:
                z_outlier_count += 1
                z_outlier_index_list.append(counter)
            counter += 1
        print(f"Z Score Method Threshold = {z_threshold}")
        print(f"Number of Outliers found: {z_outlier_count} ({(z_outlier_count/target.shape[0])*100:.3f} %)")
        
        print()
        print("Method 3 - Percentile Method:")
        print("Outliers will be located below 1st percentile and above 99th percentile.")
        first_percentile = target.quantile(0.01)
        ninenine_percentile = target.quantile(0.99)
        percentile_outlier_counter = 0
        percentile_outlier_index_list = []
        for i in range(target.shape[0]):
            if (target.iloc[i] < first_percentile) or (target.iloc[i] > ninenine_percentile):
                percentile_outlier_counter += 1
                percentile_outlier_index_list.append(i)
        print(f"1st Percentile for '{target_variable}' Column: {first_percentile:.3f}")
        print(f"99th Percentile for '{target_variable}' Column: {ninenine_percentile:.3f}")
        print(f"Number of Outliers Found {percentile_outlier_counter} ({(percentile_outlier_counter/target.shape[0]) * 100:.3f} %)")
        self.outlier_bounds.update({'Percentile Lower Bound': first_percentile, 'Percentile Upper Bound': ninenine_percentile})
        
        if iqr_outlier_count > 0:
            self._analyze_outlier_values(df, iqr_outlier_index_list, target_variable)
        
        del df, iqr_outlier_count, quantile1, quantile3, iqr, i, z_scores, z_threshold, counter, score, z_outlier_count, first_percentile, ninenine_percentile, percentile_outlier_counter, target, target_variable
        self.outlier_indices_lists = {
            'IQR Outliers Index List': iqr_outlier_index_list,
            'Z Score Outliers Index List': z_outlier_index_list,
            'Percentile Outliers Index List': percentile_outlier_index_list,
            'Outlier Bounds': self.outlier_bounds
        }
        return self.outlier_indices_lists
    def _analyze_outlier_values(
            self,
            df: pd.DataFrame,
            outliers: list[int],
            target_variable: str = 'Value',
            drop_col: list[str] = ['Phase_Date_Specification']
            ):
        """
        Will provide an overview of the characteristics for the outliers that are given for a certain dataset and column.

        Parameters:
            - df: (pd.DataFrame) DataFrame that is currently being analyzed by the program for the specified function. 
            - outliers (list[int]): A list of the outliers. This will indicate the values of the indices were the outliers are located based upon the
            method that used to obtain the list itself (IQR, Percentile Method, or Z-Score Method)
            - target_variable (string, optional): Name of the target column. Will be analyzed and searched for outliers. If outliers are found, will be
            investigated further to determine if to remove from the DataFrame.
            - drop_col (list[str], optional): Program will read through this list to drop certain columns from the DataFrame for statistical calculations.
            If the function is returning a DataFrame, the columns will be reinserted back.
        
        Returns:
            - None

            Will print out information about location of outliers based upon columns and the unique values of the columns. Based on those unique values, will print
            out how many outliers are found within each unique group within the column.
        """

        modified_df = df.copy()
        #Will retain the original copy of the columns to be implemented back within the dataset after the changes were made or calculations were completed.
        saved_columns = modified_df[drop_col]
        #Fail-Safe within the program - even if the user attempts to remove the column from the parameter, the column 'Phase_Date_Specification' will be dropped.
        if 'Phase_Date_Specification' not in drop_col:
            drop_col = drop_col + ['Phase_Date_Specification']
            saved_column = modified_df[drop_col]

        #Will drop all of the columns that are specified by the user/original parameter if parameter was not modified. This will ensure that those columns are not 
        #considered within the statistical calculations as they will not be implemented within the ML model itself as a feature.
        if len(drop_col) > 0:
            modified_df = modified_df.drop(columns=drop_col, errors='ignore')

        assert target_variable in df.columns, AssertionError; "Please identify a column that is located in the dataframe."
        isinstance(outliers, list), "Please verify that the input for 'outliers' is a list."

        copy_df = modified_df.copy()
        outliers_df = copy_df.iloc[outliers]


        print(f"Range for outliers from IQR Data for '{target_variable}' Column: {outliers_df[target_variable].min()} (min) to {outliers_df[target_variable].max()} (max)")
        print()
        print("Evaluating Outliers Based on Group Type:")
        for column in outliers_df.columns:
            if (column == 'Phase') or (column == 'Time_Period') or (column == 'Time_Period_Start_Date') or (column == 'Time_Period_End_Date') or (column == 'Value') or (column == 'Low_CI') or (column =='High_CI') or (column == 'Quartile_Range'):
                continue
            else:
                print(f"\n By {column}:")
                column_counts = outliers_df[column].value_counts()
                outlier_counts = 0
                for label, count in column_counts.items():
                    outlier_counts += count
                    print(f"'{label}': {count} Outliers")

    def apply_quartile_range_assigment(
            self, 
            df: pd.DataFrame, 
            group_cols: list[str] = ['Group', 'Subgroup', 'Phase', 'Time_Period'], 
            target_variable: str = 'Value', 
            drop_col: list[str] = ['Phase_Date_Specification'],
            use_log: bool = False
            ):
        """
        Will conduct a quartile-range assignment for the column 'Quartile_Range'. These imputations will provide a quartile range that can be considered suitable for the entry rather than
        removing the entire 'Quartile_Range' column. With healthcare statistics, any removal of data statistics can be impact the bias of the results as although the value may
        be considered an outlier, unless there is written documentation or identification or the value that was implemented was done so in error, the value may be in fact a realistic
        value for that health group and must also be considered when creating a ML model. In this case, quartile ranges can help improve the accuracy of the ML model and improve the
        results of any predictions that will come out of both of the models.

        The function will first split the dataframe (df) into two components: a mask that will identify all of the entries where 'Quartile_Range' is missing a values and the entries where there is an entry.
        For the entries where 'Quartile_Range' doesn't have a value, it will now generate a quartile-range based on the assignment of data if there is sufficient data (there must be at least 4 values within the grouping).

        Using the columns specified in the variable "group_cols", when all of the groups have found their quartiles, a Series will returned back which will be minimum and maximum values of 
        the target variable ('Value' column), and will apply the values back as a string within the 'Quartile_Range' column. Due to the various issues that may occur from insufficient data (when 
        a group may have less than 4 data values within it), and various implementing strategies that can be used, the program attempts to find the optimal grouping strategy that will maximize the
        number of data values within 1 single group through a specified hierarchy:
                - Level 1: ['Group', 'Subgroup', 'Phase', 'Time_Period'] (Will test using all of the columns originally implemented)
                - Level 2: ['Group', 'Subgroup', 'Phase'] (Will develop a quartile range using everything except the column 'Time_Period')
                - Level 3: ['Group', 'Subgroup'] (Excludes 'Phase' and 'Time_Period')
                - Level 4: ['Subgroup'] (Will only group by 'Subgroup' for quartile range assignment)
                - Level 5: ['Group'] (Will only group by 'Group' for quartile range assignment)
        
        After the program will obtain the results for each of the different levels and the various quartile ranges, the program will then test the grouping strategies and selects the one that will maximizes the percentage of
        groups with sufficient data for quartile calculation.  However, if no optimal grouping strategy was found throughout the hierarchy, a fallback option is provided where the minimum and maximum values of the target variable grouping is
        provided instead. Once chosen the program will then process the resulting group with the appropriate method. 
        
        If properly completed, the program will return the dataframe with values imputed for the column 'Quartile_Range'. 

        Parameters:
            - df (pd.DataFrame) : Dataframe that will be modified and have values imputed for the column 'Quartile_Range'. Prior to starting the imputation, the function will verify that the column exists. If the column 'Quartile_Range' is not found within the DataFrame, the function will be halted and an AssertionError will be given to the user.
            - group_cols (list[str], optional): A list of columns that will be used to first group the DataFrame prior to creating the quartile bins and continuing to create the quartiles that were found specifically for that group. 
            - drop_col (list[str], optional): Program will read through this list to drop certain columns from the dataframe for statistical calculations. If the function is returning a DataFrame, the columns will be reinserted back.
            - target_variable (str, optional): Identifies the column that signifies the target variable for the REGRESSION ML model. (Classification ML model is not considered at this time as the labels for the classification model will not be impacted by the quartile range.)
            - use_log: (bool, optional) Determines if the values within the column classified for "target_variable" will undergo a logarithmic transformation. This will assist with values that are currently considered outliers by the program.

        Returns:
            imputed_df (pd.DataFrame): An imputed dataframe which contains values for 'Quartile_Range' column.
        """

        modified_df = df.copy()
        #Will retain the original copy of the columns to be implemented back within the dataset after the changes were made or calculations were completed.
        saved_columns = modified_df[drop_col]
        #Fail-Safe within the program - even if the user attempts to remove the column from the parameter, the column 'Phase_Date_Specification' will be dropped.
        if 'Phase_Date_Specification' not in drop_col:
            drop_col = drop_col + ['Phase_Date_Specification']
            saved_columns = modified_df[drop_col]

        #Will drop all of the columns that are specified by the user/original parameter if parameter was not modified. This will ensure that those columns are not 
        #considered within the statistical calculations as they will not be implemented within the ML model itself as a feature.
        if len(drop_col) > 0:
            modified_df = modified_df.drop(columns=drop_col, errors='ignore')

        modified_df = df.copy()
        modified_df = modified_df.drop(columns=['Phase_Date_Specification'], errors='ignore')
        mask = modified_df['Quartile_Range'].isnull()
        
        def assign_labels_sufficient_data(group, use_log):
            target = group[target_variable]
            if use_log:
                log_target = np.log(target)
            else:
                log_target = target

            try:
                bins = log_target.quantile([0, 0.25, 0.5, 0.75, 1]).values
                bins[0] = bins[0] - 1e-8
                bins[-1] = bins[-1] + 1e-8

                if use_log:
                    orig_bins = np.exp(bins)
                else:
                    orig_bins = bins
                labels = [f"{orig_bins[i]:.1f} - {orig_bins[i+1]:.1f}" for i in range(len(orig_bins)-1)]
                quartiles = pd.cut(target, bins=orig_bins, labels=labels, include_lowest=True)

                del bins, use_log, log_target, labels, orig_bins
                return quartiles.astype(str).fillna(f"{target.min():.1f} - {target.max():.1f}")
            except Exception:
                return assign_labels_insufficient_data(group)
        
        def assign_labels_insufficient_data(group):
            target = group[target_variable]
            sorted_target = sorted(target.dropna()) #Will remove any final NaN values that might have been disregarded and unidentified originally.

            if len(sorted_target) == 0:
                del sorted_target
                return pd.Series([f"0.0 - 0.0"] * len(group), index=group.index)
            elif len(sorted_target) == 1:
                value = sorted_target[0]
                del sorted_target
                return pd.Series([f"{value:.1f} - {value:.1f}"] * len(group), index=group.index)
            elif len(sorted_target) == 2:
                return pd.Series([f"{sorted_target[0]:.1f} - {sorted_target[1]:.1f}"] * len(group), index=group.index)
            elif len(sorted_target) == 3:
                lower_value = sorted_target[0]
                middle_value = sorted_target[1]
                upper_value = sorted_target[2]

                result = []
                for val in target:
                    if pd.isna(val):
                        result.append(f"{lower_value:.1f} - {upper_value:.1f}") #If there is a null value in the target_variable column for that entry, the 'Quartile_Range' column is automatically filled with the upper and lower values for this entry.
                    elif val <= middle_value:
                        result.append(f"{lower_value:.1f} - {middle_value:.1f}") #If a value is located in the target_variable column and there is insufficient data for the grouping, if the value is less than or equal to "middle_value" variable, will have the quartile range show here.
                    else:
                        result.append(f"{middle_value:.1f} - {upper_value:.1f}") #If a value is located in the target_column and there is insufficient data for the grouping, if the value is greater than "middle_value" variable, will be assigned quartile range showcased here.
                
                del lower_value, middle_value, upper_value, value, target, sorted_target
                return pd.Series(result, index=group.index)
            else:
                return pd.Series([f"{sorted_target[0]:.1f}" - f"{sorted_target[-1]:.1f}"] * len(group), index=group.index)
        
        def adaptive_group_strategy(data, original_group_cols):
            """
            Adaptive grouping strategy used by the function. Main goal of this sub-function is to find the optimal grouping strategy through testing
            of different combinations and determination where the maximization of grouping occurs with sufficient data. The data subset must have at least 4 unique values.
            """
            strategies = [
                original_group_cols,
                [col for col in original_group_cols if col != 'Time_Period'],
                [col for col in original_group_cols if col not in ['Time_Period', 'Phase']],
                ['Subgroup'] if 'Subgroup' in data.columns else ['Group'],
                ['Group']
            ]

            best_strategy = original_group_cols
            best_score = -1

            for strategy in strategies:
                if not strategy:
                    continue

                try:
                    combined = data.groupby(strategy)
                    num_of_adeq_groups = 0 # Out of all of the groups, how many are actually following the rule of have at least 4 data values in it.
                    total_groups = 0 # Total number of groups that were found by the strategy.

                    for name, group in combined:
                        total_groups += 1
                        unique_values = group[target_variable].dropna().nunique()
                        if unique_values >= 4:
                            num_of_adeq_groups += 1
                    if total_groups > 0:
                        score = num_of_adeq_groups / total_groups

                        #Will compare the strategy being tested to the best strategy that is remembered by the program. If the score is better, score and strategy will be replace.
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy
                        
                            del score, strategy
                    del combined, num_of_adeq_groups, total_groups, name, unique_values, group
                except Exception:
                    continue
            del best_score, original_group_cols, data, strategies
            return best_strategy
        def implement_optimal_strategy(data, strategy):
            """
            Sub-function that will be used to process the optimal strategy that was found by the function "adaptive_group_strategy(data, original_group_cols)"
            """
            results = []
            group_iter = data.groupby(strategy, group_keys=False)

            for group_identity, group_subset in group_iter:
                num_distinct = group_subset[target_variable].dropna().nunique()

                if num_distinct >= 4:
                    group_output = assign_labels_sufficient_data(group_subset, use_log)
                else:
                    group_output = assign_labels_insufficient_data(group_subset)
                
                results.append(group_output)
            return pd.concat(results) if results else pd.Series([], dtype=str)
        if mask.any():
            incomplete = modified_df[mask].copy()
            distinct_group_types = incomplete['Group'].unique()
            final_output = []

            for current_group in distinct_group_types:
                group_subset = incomplete[incomplete['Group'] == current_group]
                best_strategy = adaptive_group_strategy(group_subset, group_cols)
                processed_output = implement_optimal_strategy(group_subset, best_strategy)
                final_output.append(processed_output)
            if final_output:
                merged_output = pd.concat(final_output)
                for row_index in merged_output.index:
                    if row_index in modified_df.index:
                        modified_df.loc[row_index, 'Quartile_Range'] = merged_output.loc[row_index]

        modified_df[drop_col] = saved_columns
        del drop_col, saved_columns, row_index, merged_output, final_output, processed_output, incomplete, distinct_group_types, current_group, group_subset, best_strategy, group_cols
        return modified_df
    def comparitive_analysis_target_values(
            self,
            df: pd.DataFrame,
            target_variable: str = 'Value',
            drop_col: list[str] = ['Phase_Date_Specification']
    ):
        """
        Will complete complete a statistical comparative analysis between the values found in the classified "target_variable" variable before the values undergo a logarithmic transformation and after they have gone through a logarithmic transformation.
        
        Parameters:
            - df (pd.DataFrame): The dataframe which contains the target column as well as the values.
            - target_variable (str, optional): Column which is identified as the target variable for the REGRESSIONAL ML MODEL ONLY. This column will have its values analyzed by the function.
            - drop_col (list[str], optional): A list of columns that is meant to be dropped by the program during statistical calculations. If the program needs to return the dataframe back to the user, will place the columns back in the dataset prior to returning the dataframe.

        Returns:
            Figure (plt.figure): A set of visualizations that will showcase a comparison between the "target_variable" values before and after undergoing a logarithmic transformation.
        """

        modified_df = df.copy()
        saved_columns = modified_df[drop_col]

        if 'Phase_Date_Specification' not in drop_col:
            drop_col = drop_col + ['Phase_Date_Specification']
            saved_columns = modified_df[drop_col]

        modified_df = modified_df.drop(columns=drop_col, errors='ignore')
    
    def display_correlation_matrix_visualization(
            self,
            df: pd.DataFrame,
            drop_col: list[str] = ['Phase_Date_Specification']
        ):
        """
        The method intends to showcase the dataframe and its current features through a correlation matrix on a Seaborn heatmap.
        Allows the researcher to be able to find the features that could be considered useful for the ML model, but also to see if there are any
        features that have possible glaring collinearity that we need to be careful about right away.

        Parameters:
            - df (pd.DataFrame): DataFrame that will be analyzed through a correlation matrix and visualized through a heatmap.
            - drop_col (list[str], optional): list of columns that need to be dropped from the DataFrame for statistical analysis.

        Returns:
            - figure (sns.heatmap): A Seaborn heatmap that will be used to analyze the correlation between all of the features that are found within the
            features within the DataFrame currently.
        """