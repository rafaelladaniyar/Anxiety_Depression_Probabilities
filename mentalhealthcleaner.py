import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import re


class MentalHealthDataCleaner:
    """
        This class is the first module of several that will be used by the project to both analyze and solve the two-part problem presented. The problem is as follows:

            (1) To label the expected condition that a certain group of the population could be experience based on certain features.
            (2) Generate the proportion of the population of that demographic group that would experience that condition.

        In this class, the set of functions will be used to conduct an overview of the dataset, load the dataset, as well as conduct an initial clean of the dataset prior to
        any indepth statistical analysis that will be done. Visualizations can also be completed through this class as well.

        The following functions are contained within this class:
            - self.load_original_dataset()
            - self.obtain_information_about_df(df, verbose_info=False, showcounts=True, reveal_columns=True, reveal_null_rates=False, explain_columns=True)
            - self.fixing_dataframe_columns(df, date_variables, drop_variables)
            - self.creating_null_rate_df(df, drop_col=['Phase_Date_Specification'])
            - self.removing_null_from_target_variable(df, target_variable='Value')
            - self.analyzing_target_variable_patterns(df, target_variable='Value', drop_col=['Phase_Date_Specification'])
    """
    def __init__(self):
        """This is all of the components that are found throughout the class. """
        self.datasetpath = "Indicators_of_Anxiety_or_Depression_Based_on_Reported_Frequency_of_Symptoms_During_Last_7_Days.csv"
        self.headerlocation = 0
        self.original_data_shape = None
        self.final_data_shape = None

    def load_original_dataset(self):
        """
        Will pull and read the dataset that the model will be attempting to analyze and use to train upon. 
        In this case, it is using the information that was obtained from the National Center for Health Statistics
        in partnership with the Census Bureau. The information was obtained from surveys that were conducted online
        from April 23, 2020 to September 16,2024. The data was last publicly updated on April 23, 2025 on cdc.gov and data.gov.
        The link can also be found here: https://catalog.data.gov/dataset/indicators-of-anxiety-or-depression-based-on-reported-frequency-of-symptoms-during-last-7- 
        as well as in the references in the notebook.

        Parameters:
            - None
        
        Returns:
            - df (pd.DataFrame): The original version of the dataset that will be implemented throughout the project.
        """
        df = pd.read_csv(self.datasetpath, header=self.headerlocation)
        self.original_data_shape = df.shape
        return df
    def obtain_information_about_df(
            self,
            df: pd.DataFrame, 
            verbose_info: bool = False, 
            showcounts: bool = True, 
            reveal_columns: bool = True, 
            reveal_null_rates: bool = False, 
            explain_columns: bool = True
            ):
        """
        Inspects and provides information about the DataFrame that is being provided. Will include basic information,
        including but not limited to number of columns, number of rows, and other factors that are inserted by the user.

        Parameters:
            - df (pd.DataFrame): DataFrame that will be inspected by the program. In the case of the project, this will be the mental-health dataset that was obtained from the "data.gov" website (can be found in the references).
            - verbose_info (bool, optional): Will provide verbose information about the original dataframe. Will showcase a list of all of the column names, how much memory it is taking on the computer, the range of entries, and more using the input line "df.info(verbose=verbose_info)" if verbose is True. Will work in conjunction with the variable "showcounts".
            - showcounts (bool, optional): Will provide information about the number of non-null values found within each column when value is equal to "True" and inputted into the code "df.info(show_counts=showcounts)". This variable is associated with the variable "verbose_info".
            - reveal_columns (bool, optional): Will provide information about the name of the columns within the dataframe if associated with the boolean value "True." Associated with the variables "reveal_null_rates" and "explain_columns."
            - reveal_null_rates (bool, optional): Will provide information about the null rates for each of the columns immediately for the user to see if associated with the boolean value "True." Associated with the variables "reveal_columns" and "explain_columns."
            - explain_columns (bool, optional): Will provide information about the specific column type for the column it is describing for the user if the variable is associated with the boolean value "True." Associated with the variables "reveal_columns" and "explain_columns."

        Returns:
            - None

            Will provide information about the DataFrame (shape, column information, column type, null information) based on user preference.
        """
        check_df = df.copy()
        column_info_dict = {}
        print(f"Information about DataFrame:")
        if (verbose_info == False) or (showcounts==True) or (reveal_columns==True) or (reveal_null_rates==False) or (explain_columns==True):
            if (verbose_info == False) & (showcounts==True):
                print(check_df.info(verbose=verbose_info, show_counts=showcounts))
                print()
            if (verbose_info == False) & (showcounts == False):
                print(check_df.info(verbose=verbose_info, show_counts=showcounts))
                print()
            if (verbose_info == True) & (showcounts == True):
                print(check_df.info(verbose=verbose_info, show_counts=showcounts))
                print()
            if (verbose_info == True) & (showcounts == False):
                print(check_df.info(verbose=verbose_info, show_counts=showcounts))
                print()
            
            print("Number of Rows:", df.shape[0])
            print("Number of Columns:", df.shape[1])
            null_rate = round((check_df.isnull().sum()/df.shape[0]) * 100, 3)
            for column in check_df.columns:
                column_info_dict[column] = {
                    'Column Name': column,
                    'Column Null Rate': f'{null_rate[column]:.3f}%',
                    'Current Column Type': check_df[column].dtype
                }
            if (reveal_columns == True) & (reveal_null_rates == True) & (explain_columns == True):
                print("Further information about Columns:")
                for column in check_df.columns:
                    print(f"Column Name: {column_info_dict[column]['Column Name']}, Null Rate: {column_info_dict[column]['Column Null Rate']}, Type: {column_info_dict[column]['Current Column Type']}")
            if (reveal_columns == True) & (reveal_null_rates == True) & (explain_columns == False):
                print("Further information about Columns:")
                for column in check_df.columns:
                    print(f"Column Name: {column_info_dict[column]['Column Name']}, Null Rate: {column_info_dict[column]['Column Null Rate']}")
            if (reveal_columns == True) & (reveal_null_rates == False) & (explain_columns == False):
                print("Further information about Columns:")
                for column in check_df.columns:
                    print(f"Column Name: {column_info_dict[column]['Column Name']}")
            if (reveal_columns == True) & (reveal_null_rates == False) & (explain_columns == True):
                print("Further information about Columns:")
                for column in check_df.columns:
                    print(f"Column Name: {column_info_dict[column]['Column Name']}, Type: {column_info_dict[column]['Current Column Type']}")
            if (reveal_columns == False) & (reveal_null_rates == True) & (explain_columns == True):
                print("Further information about Columns:")
                for column in check_df.columns:
                    print(f" {column} Null Rate: {column_info_dict[column]['Column Null Rate']}, Type: {column_info_dict[column]['Current Column Type']}")
            if (reveal_columns == False) & (reveal_null_rates == False) & (explain_columns == True):
                print("Further information about Columns:")
                for column in check_df.columns:
                    print(f"Column {column} Type: {column_info_dict[column]['Current Column Type']}")
            if (reveal_null_rates == False) & (reveal_null_rates == True) & (explain_columns == False):
                print("Further information about Columns:")
                for column in check_df.columns:
                    print(f"Column {column} Null Rate: {column_info_dict[column]['Column Null Rate']}")
            if (reveal_columns == False) & (reveal_null_rates == False) & (explain_columns == False):
                print("No further information about the columns at this time.")

    def fixing_dataframe_columns(
            self, 
            df: pd.DataFrame,
            date_variables: list[str],
            drop_variables: list[str]
            ):
        """
        Allows for any reformatting that is needed for the DataFrame. Implements the usage python library to fix the column names. 
        Will also convert columns where dates exist and change them to string. In the columns will also remove any whitespaces
        to avoid any future problems from occurring when the ML model will be reading the column names.

        Parameters:
            - df: (pd.DataFrame) the DataFrame that will be reformatted at this time.
            - date_variables (list[str]): The list of columns that need to be associated with the "datetime"-dtype. Any changes that occurred
            to the columns in the DataFrame will also occur to the names associated in this list as well. The function must be a list or else
            the function will not continue.
            - drop_variables (list[str]): The list of columns that need to be dropped by the program from the DataFrame. These columns will be
            removed first from the DataFrame prior to any changes being made, so will not be going through changes within the function as well.
            The variable must be a list type or else the function will not continue.
        """
        isinstance(drop_variables, list), "Please ensure that 'drop_variables' is a list."
        isinstance(date_variables, list), "Please ensure that 'date_variables' is a list."

        cleaned_df = df.copy()
        del df

        if (len(date_variables) > 0) or (len(drop_variables) > 0):
            if len(drop_variables) > 0:
                cleaned_df = cleaned_df.drop(columns=drop_variables, errors='ignore')
                del drop_variables
            
            before_cols = list(cleaned_df.columns)

            cleaned_df.columns = cleaned_df.columns.str.strip()
            if len(date_variables) > 0:
                for col in date_variables:
                    col = col.strip()
                    del col

            cleaned_df.columns = cleaned_df.columns.str.replace(" ", "_")
            if len(date_variables) > 0:
                for col in date_variables:
                    col = col.replace(" ", "_")
                    del col

            after_cols = list(cleaned_df.columns)
            
            del before_cols, after_cols

            if len(date_variables) > 0:
                for column in date_variables:
                    if column in cleaned_df.columns:
                        cleaned_df[column] = pd.to_datetime(cleaned_df[column], errors='coerce')
                del column, date_variables
        
        if 'Phase' in cleaned_df.columns:
            cleaned_df["Phase_Date_Specification"] = cleaned_df["Phase"].str.extract(r'\((.*?)\)')
            cleaned_df['Phase'] = cleaned_df['Phase'].str.extract(r'(\d+\.?\d*)')
        self.final_data_shape = cleaned_df.shape
        print(f"Updated Shape of the DataFrame: {self.final_data_shape}")         
        return cleaned_df
    
    def creating_null_rates(
            self, 
            df: pd.DataFrame, 
            drop_col: list[str]=['Phase_Date_Specification']
            ):
        """
        Will generate a series with the null rate for the column(s) found in the DataFrame/series.

        Parameters:
            - df: (pd.DataFrame) DataFrame or Series that will be investigated for null rates.
            - drop_col: (list[str], optional) A list of the columns within the DataFrame that need to be disregarded by the program.
        The program will retain an original copy of the columns that will be inserted back afterwards, but they will
        not be used within the calculations themselves.

        Returns:
        null_rate_df: (pd.Series) A Series which contains the null rate of all of the columns which are specified.
        """
        modified_df = df.copy()
        #Will retain the original copy of the columns to be implemented back within the dataset after the changes were made or calculations were completed.
        orig_columns = list(modified_df.columns)

        #Will drop all of the columns that are specified by the user/original parameter if parameter was not modifieds.
        if len(drop_col) > 0:
            modified_df = modified_df.drop(columns=drop_col, errors='ignore')

        #Fail-Safe within the program - even if the user attempts to remove the column from the parameter, the column 'Phase_Date_Specification' will be dropped.
        if 'Phase_Date_Specification' not in drop_col:
            modified_df = modified_df.drop(columns=['Phase_Date_Specification'], errors='ignore')

        null_rate_series = round(((modified_df.isnull().sum() / modified_df.shape[0]) * 100), 3)

        del df, modified_df, orig_columns
        return null_rate_series
    
    def removing_null_from_target_variable(
            self,
            df: pd.DataFrame,
            target_variable='Value'
            ):
        """
        The function looks through the column that is assigned as the target variable and will identify the indices
        for each of the null values that are located with in the target variable. From there, the null values will be
        removed from the DataFrame.

        Parameters
            - df (pd.DataFrame): Must be pd.DataFrame
            - target_variable (str, optional): Name of the column that is assigned as the target variable.

        Returns:
            - modified_df (pd.DataFrame): DataFrame with target variable having no values
        """
        modified_df = df.copy()

        missing_labels_indices = []
        missing_target_labels = modified_df[target_variable].isna()
        for i in range(missing_target_labels.shape[0]):
            if missing_target_labels.iloc[i] == True:
                missing_labels_indices.append(i)
        null_rate_wanted = round(((modified_df[target_variable].isnull().sum() / modified_df[target_variable].shape[0]) * 100), 3)
        null_rate_achieved = round(((len(missing_labels_indices)/modified_df[target_variable].shape[0]) * 100), 3)

        assert null_rate_wanted == null_rate_achieved, AssertionError; "There are missing null values that are unaccounted at this time. Please review your calculations."

        modified_df = modified_df.drop(index=missing_labels_indices, errors='ignore')
        modified_df.reset_index(drop=True, inplace=True)
        self.final_data_shape = modified_df.shape
        print(f"Updated DataFrame Shape: {self.final_data_shape}")
        del missing_labels_indices, missing_target_labels, i, target_variable, null_rate_wanted, null_rate_achieved, df
        return modified_df
    
    def analyzing_target_variable_patterns(
            self, 
            df: pd.DataFrame,
            target_variable: str = 'Value',
            drop_col: list[str] = ['Phase_Date_Specification']
            ):
        """
        This function will be generating a set of visualizations that will be used to analyze the DataFrame in question.
        Target variable was specifically highlighted to be used to analyze as we are expecting to find outliers and attempting
        to remove any that could result in inaccurate results within the ML model.
        The function will generate 6 visualizations in total. The 6 visualizations are as follows:
                - Visualization 1: Showcasing where the missing data is located. A heat map is used is verify where the rest of the
                missing data is found in the DataFrame.
                - Visualization 2: Horizontal bar plot is implemented to see the percent of missing data found in the column "Quartile_Range"
                found in the various groups in the column "Group".
                - Visualization 3: Histogram to show the distribution of the target column ("Value") as the ML model that will be implemented
                will be a regression model. In the top right corner will be a legend that will display the values that were calculated
                by the program as well for the column with it's mean and median (hence why the null values had to be removed prior to 
                the visualizations being created).
                - Visualization 4: Showcases the differences in the percentage of missing data in the column "Quartile_Range" once again but
                based upon the grouping conducted by the column "Indicator".
                - Visualization 5: Showcases the Average percentage of population that showcases some sort of symptoms for anxiety, depression,
                or both when solely looking at the age groups when inspecting the column "Subgroup" and inspecting the target column.
                - Visualization 6: Generates a time series graph that will depict three trendlines (one for only anxiety disorder, one for only
                depressive disorder, and one for a combination of both) with consideration to the start date of survey collection when looking at
                the trends in population percentages with symptoms of the disorders. 
       
        Parameters:
            - df (pd.DataFrame): DataFrame that is being analyzed at the time. Must already be cleaned in the target_variable prior
        to being inserted into the function.
            - target_variable (string, optional): Assigned at this time by the user, and required prior to creation of the visualizations. If
        the assertion does not pass, the visualizations will be created.
            - drop_col (list[str], optional): A list of column variables that the user can choose to drop from the table for statistical calculations and observations. If a DataFrame needs to be returned at this time, the columns will be reinserted back with the appropriate modifications if any changes were done.
        
        Returns:
            Fig (plt.figure): A combination of all six plots together which will provide the statistical analysis of the dataset.
        """
        modified_df = df.copy()
        #Will retain the original copy of the columns to be implemented back within the dataset after the changes were made or calculations were completed.
        saved_columns = modified_df[drop_col]
        #Fail-Safe within the program - even if the user attempts to remove the column from the parameter, the column 'Phase_Date_Specification' will be dropped.
        if 'Phase_Date_Specification' not in drop_col:
            drop_col = drop_col + ['Phase_Date_Specification']
            saved_column = modified_df[drop_col]

        #Will drop all of the columns that are specified by the user/original parameter if parameter was not modifieds.
        if len(drop_col) > 0:
            modified_df = modified_df.drop(columns=drop_col, errors='ignore')

        target = modified_df[target_variable]
        null_rate = round(((target.isnull().sum()/target.shape[0])*100),3)
        np.testing.assert_allclose(null_rate, 0.000, atol=1e-3), AssertionError; "Function cannot continue at this time. Please ensure your target value has no null values prior to creating your desired visualizations."


        plt.style.use('default')
        sns.set_palette('husl')
        fig, axes = plt.subplots(3, 2, figsize=(40, 40))
        fig.suptitle('Comprehensive Analysis for All \n of the Features with the DataFrame', fontsize=36, fontweight='bold', fontstyle='oblique')
    
        ax1 = axes[0, 0]
        missing_data = modified_df.isnull().sum()
        null_rates_for_df = (missing_data / modified_df.shape[0]) * 100

        missing_matrix = modified_df.isnull().astype(int)
        sns.heatmap(missing_matrix.T, cmap='Reds', cbar=True, ax=ax1,
                    yticklabels=modified_df.columns, xticklabels=False)
        ax1.set_title('Missing Values and Their Patterns Found Across Dataset \n (Dark Red = Missing)',
                      fontsize=20, fontweight='bold')
        ax1.set_xlabel('Dataset Record Location')

        ax2 = axes[1, 0]
        quartile_missing = modified_df.groupby('Group')['Quartile_Range'].apply(lambda x: x.isnull().mean() * 100)
        quartile_missing = quartile_missing.sort_values(ascending=True)

        bars = ax2.barh(range(len(quartile_missing)), quartile_missing.values, color='orange')
        ax2.set_title('Quartile Range Missing Data % By Group', fontsize=20, fontweight='bold')
        ax2.set_xlabel('Missing Data Percentages (%)')
        ax2.set_yticks(range(len(quartile_missing)))
        ax2.set_yticklabels(quartile_missing.index, fontsize=10)
        for i, v in enumerate(quartile_missing.values):
            ax2.bar_label(bars, label_type='center', fmt='%.3f')

        ax3 = axes[2, 0]
        ax3.hist(target, bins=30, color='orange', alpha=0.8, edgecolor='black')
        ax3.axvline(target.mean(), color='red', linestyle='--', label=f'Mean: {target.mean():.2f}')
        ax3.axvline(target.median(), color='purple', linestyle='--', label=f'Median: {target.median():.2f}')
        ax3.set_title('Distribution of Mental Health Values \n (Based on Population Percentages)', fontsize=20, fontweight='bold')
        ax3.set_xlabel('Symptom Percentage (%)')
        ax3.set_ylabel('Frequency')
        ax3.legend()

        ax4 = axes[0, 1]
        indicator_missing = modified_df.groupby('Indicator')['Quartile_Range'].apply(lambda x: x.isnull().mean() * 100)
        colors = ['red', 'orange', 'darkred']
        bars = ax4.bar(range(len(indicator_missing)), indicator_missing.values, color=colors)
        ax4.set_title('Missing Quartile Ranges Based \n On Mental Health Indicators', fontsize=20, fontweight='bold')
        ax4.set_ylabel('Missing Data Percentages (%)')
        ax4.set_xticks(range(len(indicator_missing)))
        labels = [ind.replace(' ', '\n') for ind in indicator_missing.index]
        ax4.set_xticklabels(labels, fontsize=9)
        for i, v in enumerate(indicator_missing.values):
            ax4.text(i, v, f'{v:.3f}%', ha='center', fontweight='bold')
        
        ax5 = axes[1, 1]
        age_data = modified_df[modified_df['Group'] == 'By Age']
        if len (age_data) > 0:
            age_avg = age_data.groupby('Subgroup')['Value'].mean().sort_values(ascending=False)
            x_pos = range(len(age_avg))
            ax5.bar(x_pos, age_avg.values, color='lightgreen', alpha=0.8)
            ax5.set_title('Average Percent of Population with Symptoms \n of Anxiety or Depression by Age Group', fontsize=20, fontweight='bold')
            ax5.set_ylabel('Average Symptom Percentage (%)')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(age_avg.index, rotation=45, ha='right', fontsize=9)
            for i,v in enumerate(age_avg.values):
                ax5.text(i, v, f'{v:.3f}%', ha='center', fontsize=8)

        ax6 = axes[2, 1]
        modified_df['Start_Date']= pd.to_datetime(modified_df['Time_Period_Start_Date'], errors='coerce')
        time_trend = modified_df.groupby(['Start_Date', 'Indicator'])['Value'].mean().reset_index()

        indicators = time_trend['Indicator'].unique()
        colors = ['blue', 'red', 'purple']
        for i, indicator in enumerate(indicators):
            indicator_data = time_trend[time_trend['Indicator'] == indicator]
            ax6.plot(indicator_data['Start_Date'], indicator_data['Value'],
                     marker='o', label=indicator, color=colors[i % len(colors)], alpha=0.7)
        ax6.set_title('Mental Health Trends Over Time With \n Consideration to Start Date', fontsize=20, fontweight='bold')
        ax6.set_xlabel('Start Date of Survey Data Collection')
        ax6.set_ylabel('Population Percentage Showing Symptoms (%)')
        ax6.legend(fontsize=8, loc='upper right')
        ax6.tick_params(axis='x', rotation=45)

        
        plt.show()
        del target, target_variable, ax1, ax2, axes, ax3, ax4, ax5, ax6, missing_data, missing_matrix, null_rate, null_rates_for_df, df, colors, bars, indicator_missing, indicator_data, indicator, quartile_missing, i, v, age_data, x_pos, age_avg, time_trend, labels, modified_df, saved_columns, drop_col
        return fig
                