

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


def make_simulation_df(num_rows, pheno_list = ['A','B','C']):
    import numpy as np
    import pandas as pd
    # Create a random array of x values between 1 and 10
    x = np.random.randint(1, 10**5, size=num_rows)
    
    # Create a random array of y values between 1 and 10
    y = np.random.randint(1, 10**5, size=num_rows)
    
    # Create a random array of phenotype values
    phenotypes = np.random.choice(pheno_list, size=num_rows)
    
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
    test= ''
    # Loop through each category column
    for category_col in category_cols:
        # Loop through each numeric column
        for numeric_col in numeric_cols:
            print(category_col , numeric_col)
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
                test = 't-test'
            # If there are more than 2 unique values in the category column
            else:
                # Perform ANOVA-oneway test
                test_result = stats.f_oneway(*[df[df[category_col] == cat][numeric_col] for cat in unique_cats])
                test = 'ANOVA oneway'
            
            # Store the test result in the test_results dictionary
            test_results[(category_col, numeric_col,test)] = test_result
            
    # Return the test results as a DataFrame
    test_results_df = pd.DataFrame(test_results, index=['test_statistic', 'p_value']).T
    
    return test_results_df



def plot_signif_cat_num(data, signif_table, plot = 'box', threshold = 0.05, figsize=(10, 5)):

    import itertools
    from scipy.stats import f_oneway, ttest_ind, zscore
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns
    import matplotlib.pyplot as plt

    signif_data = signif_table[signif_table['p_value'] < threshold]

    # Add markers for significant p-values
    for index, row in signif_data.iterrows():
        try:
            category = row['level_0']
            value = row['level_1']
            plt.figure(figsize=figsize)
            # Create the violin plot
            if plot in ['box','boxplot']:
                g = sns.boxplot(x=category, y=value, data=data)
            elif plot == 'violin':
                g = sns.violinplot(x=category, y=value, data=data)
            ax = sns.swarmplot(x=category, y=value, data=data, color=".25")

            g.set_title('pVal: {}, test: {}'.format( round(row['p_value'], 4),row['level_2'] ))
            plt.xticks(rotation=90)
            plt.show()
            # plt.plot([category-0.2, category+0.2], [value, value], linewidth=3, color='black')
        except:
            pass



def test_significance_num(df, numerical_cols, test_name):
    """
    Perform a selected statistical test on multiple numerical columns of a given DataFrame
    and return a table of pairs of variables and their significance.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the numerical columns to be tested.
        numerical_cols (list): A list of column names to be used as the numerical variables.
        test_name (str): The name of the statistical test to be performed.
    
    Returns:
        pd.DataFrame: A table of pairs of variables and their significance.
    """
    from scipy.stats import pearsonr, spearmanr, kendalltau

    # Initialize an empty list to store the test results
    test_results = []
    
    # Loop through all pairs of numerical columns and perform the selected test
    for i in range(len(numerical_cols)):
        for j in range(i+1, len(numerical_cols)):
            try:
                col1 = numerical_cols[i]
                col2 = numerical_cols[j]
                
                # Calculate correlation coefficients and p-values
                pearson_r, pearson_p = pearsonr(df[col1], df[col2])
                spearman_rho, spearman_p = spearmanr(df[col1], df[col2])
                kendall_tau, kendall_p = kendalltau(df[col1], df[col2])
                
                # Return results as a dictionary
                result = {'V1': col1,'V2': col2,
                        'Pearson_p': pearson_p,
                        'Spearman_rho': spearman_rho, 'Spearman_p': spearman_p,
                        'Kendall_tau': kendall_tau, 'Kendall_p': kendall_p}
                # print(result)
                # Append the test results to the list
                test_results.append([col1, col2, pearson_p, spearman_rho,  spearman_p, kendall_tau, kendall_p])
            except:
                pass
    print(test_results)
    # Convert the list of test results to a DataFrame and return it
    return pd.DataFrame(test_results, columns=['Variable 1', 'Variable 2', 'Pearson_p', 'Spearman_rho', 'Spearman_p', 'Kendall_tau', 'Kendall_p'])


def plot_correlation_heatmap(corr_df, alpha=0.05, figsize=[10,10]):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    Plot a heatmap of the significant correlations between numerical variables in a given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the numerical columns to be tested.
        numerical_cols (list): A list of column names to be used as the numerical variables.
        test_name (str): The name of the statistical test to be performed.
        alpha (float): The significance level to use for determining significance.
    
    Returns:
        None
    """
    # Filter the correlations that are statistically significant
    sig_corr = corr_df[(corr_df['Pearson_p'] < alpha) | (corr_df['Spearman_p'] < alpha) | (corr_df['Kendall_p'] < alpha)]
    
    # Create a pivot table of the significant correlations
    corr_pivot = sig_corr.pivot(index='Variable 1', columns='Variable 2', values='Spearman_rho')
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    sns.heatmap(corr_pivot, cmap='coolwarm')
    plt.title(f"Significant correlations")
    plt.show()

    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    sns.clustermap(corr_pivot.fillna(corr_pivot.median().median()), cmap='coolwarm')
    plt.title(f"Significant correlations")
    plt.show()

    return corr_pivot

def plot_correlation_heatmap_P_Val(corr_df, pVal_col= 'P-Value', alpha=0.05, figsize=[10,10]):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    Plot a heatmap of the significant correlations between numerical variables in a given DataFrame.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the numerical columns to be tested.
        numerical_cols (list): A list of column names to be used as the numerical variables.
        test_name (str): The name of the statistical test to be performed.
        alpha (float): The significance level to use for determining significance.
    
    Returns:
        None
    """
    # Filter the correlations that are statistically significant
    sig_corr = corr_df[(corr_df[pVal_col] < alpha) ]
    
    # Create a pivot table of the significant correlations
    corr_pivot = sig_corr.pivot(index='Variable 1', columns='Variable 2', values=pVal_col)
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    sns.heatmap(corr_pivot, cmap='coolwarm')
    plt.title(f"Significant correlations")
    plt.show()


    # Plot the heatmap
    sns.clustermap(corr_pivot.fillna(corr_pivot.median().median()), cmap='coolwarm', figsize=figsize)
    plt.title(f"Significant correlations")
    plt.show()

    return corr_pivot


def correlation_matrix(corr_df, alpha=0.05, figsize=[10,10]):
    import seaborn as sns
    import matplotlib.pyplot as plt
    sig_corr = corr_df[(corr_df['Pearson_p'] < alpha) | (corr_df['Spearman_p'] < alpha) | (corr_df['Kendall_p'] < alpha)]
    corr_pivot = sig_corr.pivot(index='Variable 1', columns='Variable 2', values='Spearman_rho')

    return corr_pivot

def plot_correlations(df, correlation_df, pVal_name=  'Pearson_p',  threshold=0.05 , clr = 'red'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    Create scatterplots and linear regression lines between pairs of variables in a DataFrame
    where the p-value of their correlation is less than a specified threshold.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the numerical variables.
        correlation_df (pd.DataFrame): A table of pairs of variables and their p-values.
        pVal_name=  'Pearson_p' | 'Spearman_p' | Kendall_p
        threshold (float): The maximum p-value for a correlation to be considered significant.
    
    Returns:
        None: Displays the scatterplots and regression lines.
    """
    # Filter the correlation DataFrame by the specified threshold
    significant_corrs = correlation_df[correlation_df[pVal_name] < threshold]
    
    # Loop through all pairs of significant correlations and create scatterplots and regression lines
    for i, row in significant_corrs.iterrows():
        var1, var2, pval = row[0], row[1], row[pVal_name]
        sns.lmplot(data=df, x=var1, y=var2, line_kws={'color': clr})
        plt.title(f'{var1} vs. {var2} (p={pval:.3f})')
        plt.show()



def test_significance_cat(df, category_cols):
    import pandas as pd
    import scipy.stats as stats
    """
    Perform a chi-squared test or a Fisher's exact test on pairs of categorical columns in a given DataFrame
    and return a table of pairs of variables and their significance.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the categorical columns to be tested.
        category_cols (list): A list of column names to be used as the categorical variables.
    
    Returns:
        pd.DataFrame: A table of pairs of variables and their significance.
    """
    # Initialize an empty list to store the test results
    test_results = []
    
    # Loop through all pairs of categorical columns and perform the appropriate test
    for i in range(len(category_cols)):
        for j in range(i+1, len(category_cols)):
            col1 = category_cols[i]
            col2 = category_cols[j]
            
            # Create a contingency table of the two categorical variables
            contingency_table = pd.crosstab(df[col1], df[col2])
            
            # Check the size of the contingency table
            n_rows, n_cols = contingency_table.shape
            if n_rows == 2 and n_cols == 2:
                # If the table is 2x2, perform Fisher's exact test
                oddsratio, pvalue = stats.fisher_exact(contingency_table)
            else:
                # Otherwise, perform the chi-squared test
                chi2, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Return results as a dictionary
            result = {'V1': col1,'V2': col2, 'P-Value': pvalue}
            
            # Append the test results to the list
            test_results.append([col1, col2, pvalue])
    
    # Convert the list of test results to a DataFrame and return it
    return pd.DataFrame(test_results, columns=['Variable 1', 'Variable 2', 'P-Value'])


def plot_cat_diff(df, significance_df, threshold=0.05, cmap="vlag"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    Visualize the differences between all categorical columns in the given DataFrame
    based on the significance results from `test_significance_cat` function.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the categorical columns to be compared.
        significance_df (pd.DataFrame): A table of pairs of variables and their significance, obtained from `test_significance_cat` function.
        threshold (float, optional): The significance threshold to use for plotting. Default is 0.05.

    Returns:
        None.
    """
    # Filter the pairs of variables with significant differences based on the threshold
    sig_pairs = significance_df.loc[significance_df['P-Value'] <= threshold, ['Variable 1', 'Variable 2']]
    
    # Loop through all pairs of categorical columns and plot the differences
    for i in range(len(df.columns)):
        for j in range(i+1, len(df.columns)):
            col1 = df.columns[i]
            col2 = df.columns[j]
            
            # Check if the pair of variables is significant
            if ((col1 in sig_pairs.values[:,0]) and (col2 in sig_pairs.values[:,1])) or ((col1 in sig_pairs.values[:,1]) and (col2 in sig_pairs.values[:,0])):
                # Create a frequency table of the two columns
                table = pd.crosstab(df[col1], df[col2])
                
                # Plot the frequency table as a heatmap using seaborn
                ax = sns.heatmap(table, cmap=cmap, annot=True, fmt='d', cbar=False, cbar_kws={"label": "Correlation coefficient"})

                # Set plot title and axis labels
                plt.title(f"{col1} vs {col2}")
                plt.xlabel(col2)
                plt.ylabel(col1)
                
                # Show the plot
                plt.show()
                


def plot_network(coordinates, edges, col_type, size_nodes=10, x = 'x', y='y', cmap_nodes='Spectral', edges_color='gray'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    """
    Plot a network using seaborn based on node coordinates, edges, and node types.
    
    Parameters:
        coordinates (pd.DataFrame): A DataFrame containing the x, y, and type columns for each node.
        edges (list of tuples): A list of tuples containing the source and target indices for each edge.
        col_type (str): The name of the column in the coordinates DataFrame that contains the node types.
        size_nodes (int): The size of the nodes in the plot.
        cmap_nodes (str): The name of the colormap to use for the node types.
        edges_color (str): The color to use for the edges.
    """

    # Create a dictionary mapping node types to integers
    type_dict = {t: i for i, t in enumerate(coordinates[col_type].unique())}
    
    # Create a list of node colors based on the node types
    node_colors = coordinates[col_type].map(type_dict)
    
    # Create a dictionary mapping node indices to coordinates
    pos_dict = {i: (coordinates.loc[i, x], coordinates.loc[i, y]) for i in range(len(coordinates))}
    print(pos_dict)
    # Create the plot using seaborn
    sns.set_style("white")
    # sns.set(rc={'axes.facecolor': 'lightgray', 'figure.facecolor': 'white'})
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.scatterplot(x=x, y=y, hue=node_colors, style=node_colors, palette=cmap_nodes, s=size_nodes,
                    edgecolor='black', linewidth=0.5, alpha=0.8, data=coordinates, ax=ax)
    
    # Draw the edges on the plot
    for edge in edges.iterrows():
        source = pos_dict[edge[1][0]]
        target = pos_dict[edge[1][1]]
        ax.plot([source[0], target[0]], [source[1], target[1]], color=edges_color, linewidth=0.5)
        
    # Add a legend to the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, type_dict.keys(), title=col_type, loc='upper right', bbox_to_anchor=(1.15, 1))
        
    # remove x and y tick labels
    plt.xticks([])
    plt.yticks([])
    # remove x and y axes
    sns.set(style="ticks", rc={"axes.grid":False})
    # sns.despine(bottom=True, left=True)

    plt.xlabel('')
    plt.ylabel('')

    # Set the axis labels and title
    ax.set_title('Network Plot')
    plt.show()



import pandas as pd
import numpy as np

def calculate_edges_dist(coords, edges, x = 'Pos_X', y = 'Pos_Y'):
    # Compute the distance between connected nodes
    distances = []
    for _, edge in edges.iterrows():
        source = edge['source']
        target = edge['target']
        source_coord = coords.loc[source, [x,y]].values
        target_coord = coords.loc[target, [x,y]].values
        distance = np.linalg.norm(target_coord - source_coord)
        distances.append(distance)

    # Add distances as a new column in the edges DataFrame
    edges['distance'] = distances
    return edges

def filter_edges_dist(edges, threshold_p = -1, threshold = 20, dist_col = 'distance'):
    """
    threshold (float): The threshold distance above which edges are removed.
    # threshold_p: threshold percentage of data
    # threshold: threshold value in micrometer
    """
    # Remove edges with a distance above the threshold
    if threshold_p > 0:
        threshold = df['values'].quantile(threshold_p)
    edges = edges[edges[dist_col] <= threshold]

    return edges