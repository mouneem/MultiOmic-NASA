

import numpy as np
import itertools
from scipy.stats import f_oneway, ttest_ind, zscore
import pandas as pd
import scipy.stats as stats
import numpy as np
import pandas as pd

def load():
    import numpy as np
    import pandas as pd


def make_simulation_df(num_rows):
    import numpy as np
    import pandas as pd
    # Create a random array of x values between 1 and 10
    x = np.random.randint(1, 10**5, size=num_rows)
    
    # Create a random array of y values between 1 and 10
    y = np.random.randint(1, 10**5, size=num_rows)
    
    # Create a random array of phenotype values
    phenotypes = np.random.choice(['A', 'B', 'C'], size=num_rows)
    
    # Create a dictionary with the data
    data = {'x': x, 'y': y, 'phenotype': phenotypes}
    
    # Create a dataframe from the dictionary
    df = pd.DataFrame(data)
    
    return df

def change_annotation(input_df, column_to_change, annotation_df, column_to_change_in_annotation_df, column_to_change_with):

    # Replaces phenotype annotations from the input dataframe by the corresponding annotations (levels) from the annotation dataframe.
    input_df[column_to_change] = input_df[column_to_change].map(annotation_df.set_index(column_to_change_in_annotation_df)[column_to_change_with])
  
    return input_df


def generalize_infrequent_phenotypes(df, annotation_df, specific_phenotype, specific_phenotype_in_annotation, desired_phenotype_level, threshold=0.05):

    """
    Replaces infrequent values of a specific phenotype column in a DataFrame with more general ones from an annotation DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        annotation_df (pandas.DataFrame): The DataFrame containing the annotation information.
        specific_phenotype (str): The name of the column in `df` that contains the phenotype to modify.
        desired_phenotype_level (str): The name of the column in `annotation_df` that contains the desired, more general phenotype.
        threshold (float, optional): The frequency threshold below which a phenotype is considered infrequent. Defaults to 0.05.
    
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    
    # Compute the frequency of each phenotype in df
    freq = df[specific_phenotype].value_counts(normalize=True)
    
    # Identify the phenotypes with frequency less than the threshold
    infrequent_phenotypes = freq[freq < threshold].index
    
    # Create a mapping between infrequent phenotypes and their equivalent in annotation_df
    mapping = annotation_df.set_index(specific_phenotype_in_annotation)[desired_phenotype_level].to_dict()
    
    # Replace infrequent phenotypes in df with their equivalent from annotation_df
    mask = df[specific_phenotype].isin(infrequent_phenotypes)
    df.loc[mask, specific_phenotype] = df.loc[mask, specific_phenotype].replace(mapping)
    
    return df


def drop_infrequent_phenotypes(df, column, threshold = 0.05):
    
    # Compute the frequency of each value in the specified column
    value_counts = df[column].value_counts(normalize=True)
    
    # Identify the values that occur less frequently than 4%
    values_to_drop = value_counts[value_counts < threshold].index.tolist()
    
    # Drop the rows that contain those values
    filtered_df = df[~df[column].isin(values_to_drop)]
    
    return filtered_df

import pandas as pd
import random
import os


def make_simulation_files(dir_path, num_files, min_rows = 5, max_rows = 10000, ROI = False):
    # The function first checks if the directory specified in dir_path exists, and creates it if it doesn't. 
    # It then loops through the number of files specified in num_files, and generates random values for the three columns using the random module.
    """" args 
    dir_path: The path to the directory where the CSV files will be created.
    num_files: The number of CSV files to create.
    num_rows: The number of rows to create in each CSV file.
    """

    # Create the directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Loop through the number of files to create
    for i in range(num_files):
        # if  min_rows = -1 => random
        num_rows = random.randint(min_rows, max_rows)


        # Generate random values for the three columns
        x_vals = [random.random() for j in range(num_rows)]
        y_vals = [random.random() for j in range(num_rows)]
        phenotype_vals = [random.choice(['A', 'B', 'C']) for j in range(num_rows)]
        ROIs = [random.choice(['ROIs_1', 'ROIs_2']) for j in range(num_rows)]
        
        # Create a dataframe from the values
        df = pd.DataFrame({
            'x': x_vals,
            'y': y_vals,
            'phenotype': phenotype_vals,
            'ROIs': ROIs
        })
        
        # Write the dataframe to a CSV file
        file_path = os.path.join(dir_path, f'simulated_{i+1}.csv')
        df.to_csv(file_path, index=False)


def coords_to_df(dir_path, keep_ext_filename = False):

    # List of CSV file names in directory
    csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]

    # Initialize empty list to store dataframes
    dfs = []

    # Loop through CSV files and append to dfs list
    for file in csv_files:
        # Read CSV file into dataframe
        df = pd.read_csv(os.path.join(dir_path, file))

        # Add filename as a new column
        if not keep_ext_filename:
            # remove ext (.csv)
            df['filename'] = os.path.splitext(file)[0]
        else:
            df['filename'] = file
        
        # Append dataframe to dfs list
        dfs.append(df)

    # Concatenate dataframes in dfs list into one dataframe
    combined_df = pd.concat(dfs, ignore_index=True)
    
    return combined_df


def split_df_roi(df, out_path, ROIs_col_name = 'ROIs', file_name = 'coordinates_' ):
    # Split the dataframe based on the value of the ROI column
    for roi_value, group_df in df.groupby(ROIs_col_name):
        # Construct the output filename
        output_filename = f"{file_name}_{roi_value}.csv"
        output_filepath = out_path / output_filename
        
        # Export the group dataframe to a new file
        group_df.to_csv(output_filepath, index=False)


def split_DIR_roi(in_path, out_path, ROIs_col_name = 'ROIs'):
    # Loop through each file in the directory
    for file_name in os.listdir(in_path):
        # Check if the file is a CSV file
        df = pd.read_csv(in_path / file_name)
        split_df_roi(df,out_path, ROIs_col_name = 'ROIs', file_name = file_name)



def test_category_numeric(df, category_cols, numeric_cols):
    import numpy as np
    import itertools
    from scipy.stats import f_oneway, ttest_ind, zscore
    import pandas as pd
    import scipy.stats as stats
    # Initialize an empty dictionary to store the test results
    test_results = {}
    
    # Loop through each category column
    for category_col in category_cols:
        # Loop through each numeric column
        for numeric_col in numeric_cols:
            # Get the unique values in the category column
            unique_cats = df[category_col].unique()
            
            # If there are only 2 unique values in the category column
            if len(unique_cats) == 2:
                # Perform T-test or Z-test based on the sample size of the numeric column
                if len(df[numeric_col]) < 30:
                    test_result = stats.ttest_ind(df[df[category_col] == unique_cats[0]][numeric_col], 
                                                  df[df[category_col] == unique_cats[1]][numeric_col])
                else:
                    test_result = stats.ttest_ind(df[df[category_col] == unique_cats[0]][numeric_col], 
                                                  df[df[category_col] == unique_cats[1]][numeric_col],
                                                  equal_var=False)
            # If there are more than 2 unique values in the category column
            else:
                # Perform ANOVA-oneway test
                test_result = stats.f_oneway(*[df[df[category_col] == cat][numeric_col] for cat in unique_cats])
            
            # Store the test result in the test_results dictionary
            test_results[(category_col, numeric_col)] = test_result
            
    # Return the test results as a DataFrame
    test_results_df = pd.DataFrame(test_results, index=['test_statistic', 'p_value']).T
    
    return test_results_df



def plot_signif_cat_num(data, signif_table, plot = 'box', threshold = 0.05):

    import itertools
    from scipy.stats import f_oneway, ttest_ind, zscore
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns

    signif_data = signif_table[signif_table['p_value'] < threshold]

    # Add markers for significant p-values
    for index, row in signif_data.iterrows():
        category = row['level_0']
        value = row['level_1']

        # Create the violin plot
        if plot in ['box','boxplot']:
            sns.boxplot(x=category, y=value, data=data)
        elif plot == 'violin':
            sns.violinplot(x=category, y=value, data=data)
            plt.show()
        # plt.plot([category-0.2, category+0.2], [value, value], linewidth=3, color='black')