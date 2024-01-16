import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import re
import tqdm

def load():
    print('Loaded !')

# Filter a DataFrame based on given conditions.
def qc_filter(qc_file, filter_dict, output_type='all', sep = '\t'):
    """ args:
    - qc_dataframe (pd.DataFrame): The input DataFrame.
    - filter_dict (dict): A dictionary where keys are column names, and values are conditions for filtering.
    - output_type (str, optional): Specify 'all' to return the entire DataFrame, or provide a specific column name to return only that column. Default is 'all'.
    """
    qc_file = qc_file
    qc_dataframe = pd.read_csv( qc_file , sep = sep)
    # Apply filters to the DataFrame
    nrows = qc_dataframe.shape[0]
    for column, values in filter_dict.items():
        qc_dataframe = qc_dataframe[qc_dataframe[column].isin(values)]

    print( 'rows selected', qc_dataframe.shape[0], 'from', nrows )
    if output_type == 'all':
        return qc_dataframe
    elif output_type in qc_dataframe.columns:
        return qc_dataframe[output_type]
    
# select random
def get_random_from_repo(folder_path, filename_pattern):
    """
    Get a random file from a folder that matches a given regular expression in the file name.
    Parameters:
    - folder_path (str): The path to the folder containing files.
    - regex_in_file_name (str): The regular expression to match in the file name.
    """
    # Get a list of files in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    matching_files = [f for f in files if re.search(filename_pattern, f)]

    # If there are matching files, randomly select one
    if matching_files:
        random_file = random.choice(matching_files)
        print("Sample selected : ",random_file)
        return os.path.join(folder_path, random_file)
    else:
        print(f"No files matching the regular expression '{filename_pattern}' found in the folder.")
        return None


# Function to check if a column is numerical
def is_numerical(column):
    return pd.api.types.is_numeric_dtype(column)

# Vizualisation of image:
def viz_sample(sample_path, xcol = 'x', ycol = 'y', colorby = ['phenotype'], by_region_dict = False, s = 50, alpha= .5, export_file = False, figure_size=(10, 8), title = 'Sample Visualization', sep = '\t', showBarplot = True, plotFacet = False):
    """
    Visualize samples based on coordinates and color-coded by specified columns.
    Parameters:
    - coords_df (pd.DataFrame): The DataFrame containing coordinates and other columns.
    - xcol (str): The column representing the x-axis coordinates.
    - ycol (str): The column representing the y-axis coordinates.
    - colorby (list): A list of column names for color-coding the samples.
    - export_file (str, optional): File path for exporting the visualization. Default is None (no export).
    """
    coords_df = pd.read_csv(sample_path, sep=sep)
    
    display(coords_df.head())
    print(coords_df.shape)
    sample_name = os.path.basename(sample_path)
    print(sample_name)
    # Set up the figure and axis
    # plt.figure(figsize=(figure_size))
    
    # Scatter plot with color-coded samples
    for colorby_element in colorby:
        coords_r = coords_df.copy()
        if by_region_dict:
            if plotFacet:
                col_var = list(by_region_dict.keys())[0]
                sns.scatterplot(data=coords_r, x=xcol, y=ycol, hue=col_var, size=s, alpha=alpha)
                plt.show()
                # Set up the facet grid
                display(coords_r.head())
                coords_r[colorby_element].fillna('Unknown')
                coords_r[colorby_element] = coords_r[colorby_element].astype('category')
                g = sns.FacetGrid(coords_r, col=col_var, hue=colorby_element, col_wrap=2)
                # Plot scatterplot in each facet
                g.map(sns.scatterplot, xcol, ycol, size=s, alpha=alpha)
                plt.show()
            else:
                for colname, regions in by_region_dict.items():
                    for region in regions:
                        fig, ax = plt.subplots(figsize = figure_size)
                        print(colname, region)
                        idx = coords_df[colname] == region
                        coords_r = coords_df.loc[idx,:]
                        print(region , coords_r.shape)
                        scatter = sns.scatterplot(data=coords_r, x=xcol, y=ycol, hue=colorby_element, size=s, alpha=alpha, ax=ax)
                        ax.set_title(title + ' Region: ' + str(region)  )
                        plt.show()
            for colname, regions in by_region_dict.items():
                for region in regions:
                    if showBarplot: 
                        barplot_prop(coords_r, phenotype_col = colorby_element, title = str(region) )
        else:
            fig, ax = plt.subplots(figsize = figure_size)
            scatter = sns.scatterplot(data=coords_r, x=xcol, y=ycol, hue=colorby_element, size=s, alpha=alpha, ax=ax)
            # Set plot labels and title
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.set_title(title)
            # Add legend
            ax.legend()
            # Show the plot
            plt.show()
            if showBarplot: 
                barplot_prop(coords_r, phenotype_col = colorby_element)
    # Export the plot if export_file is specified
    if export_file:
        plt.savefig(export_file)
        print(f"Visualization exported to: {export_file}")

# cell proportion
def barplot_prop(coords, phenotype_col, normalize = True, fig_size= (15, 5) , title = ''):
        p = coords[phenotype_col].value_counts(normalize=normalize)
        plt.figure(figsize=fig_size)
        p = sns.barplot(x=p.index, y=p.values)
        p.set_title(str(title)  )
        plt.show()
        return p

def remove_column(df, column_name):
    if column_name in df.columns:
        df = df.drop(column_name, axis=1)
    else:
        print(column_name, 'Not found!')
    return df

# Count the occurrences of each phenotype in a DataFrame and visualize the counts.
def count_phenotype(path, phenotype_col, normalize=True, export_count_path=False , sep = ',' , select_roi = False):
    """
    Parameters:
    - path (str): The path to the input file containing the DataFrames.
    - phenotype_col (str): The column name representing phenotypes.
    - normalize (bool, optional): If True, normalize the counts to percentages. Default is True.
    - export_coords_path (str, optional): File path for exporting the coordinates DataFrame. Default is False.
    - select_roi (list) colname | value to select
    """
    phenotypes_counts = []
    # for filename in os.listdir(path) :
    files = os.listdir(path)
    for filename in tqdm.tqdm(files, desc='Reading coordinate files...'):
        # Read the DataFrame from the input file
        df = pd.read_csv(os.path.join(path, filename), sep = sep)  # You can adjust this based on the actual file format
        if select_roi:
            idx = df[select_roi[0]] == select_roi[1]
            df = df.loc[idx,:]
        # Count occurrences of each phenotype
        if  isinstance(phenotype_col, (int)):
            phenotypes_count = df.iloc[:,phenotype_col].value_counts(normalize=normalize)
            pass
        else:
            phenotypes_count = df[phenotype_col].value_counts(normalize=normalize)
        phenotypes_count['filename'] = filename
        phenotypes_counts.append(phenotypes_count)
        

    phenotypes_counts = pd.DataFrame(phenotypes_counts)
    phenotypes_counts.index = phenotypes_counts['filename']
    del phenotypes_counts['filename']
    # Export coordinates DataFrame if specified
    if export_count_path:
        phenotypes_counts.to_csv(export_count_path, index=False)
    return phenotypes_counts

# Plot a heatmap from a DataFrame using seaborn.
def plot_heatmap(count_matrix, figsize=[20, 6], export_fig=False, cluster_row=True, cluster_col=True, fillna=0, cmap='Spectral', font_scale = 1, title = 'Heatmap', linewidths=0, clinical = False, merge_column_inClinical = 'Patient', col_toRemove= [''], merge_column_inCount='patient', extract_sample_pattern=False, extract_patient_pattern=False , method = 'average'):
    """
    Parameters:
    - count_matrix (pd.DataFrame): The input DataFrame.
    - figsize (list, optional): Figure size [width, height]. Default is [20, 6].
    - export_fig (str or bool, optional): File path for exporting the figure. Default is False (no export).
    - cluster_row (bool, optional): Whether to cluster rows. Default is True.
    - cluster_col (bool, optional): Whether to cluster columns. Default is True.
    - fillna (int or float, optional): Value to fill NaN entries in the DataFrame. Default is 0.
    - cmap (str, optional): Colormap to use. Default is 'Spectral'.
    """
    # Fill NaN entries in the DataFrame
    count_matrix_filled = count_matrix.fillna(fillna)
    
    # Create a heatmap using seaborn
    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)  # Adjust font size if needed
    if isinstance(clinical, pd.DataFrame):
        CountMatrix_clinical = merge_count_clinical(count_matrix_filled, clinical, merge_column_inClinical=merge_column_inClinical, merge_column_inCount= merge_column_inCount, extract_sample_pattern=extract_sample_pattern, extract_patient_pattern=extract_patient_pattern)
        clinical_features = clinical.columns
        img_features = count_matrix_filled.columns
        
        row_colors = {}
        sns.set(font_scale=0.8) 

        count_feat = CountMatrix_clinical[img_features]
        count_feat.index = CountMatrix_clinical[merge_column_inCount]
        
        clinical_feat = CountMatrix_clinical[clinical_features]


        for label in clinical_feat.columns:
            values = clinical_feat[label]
            u_vals = values.unique()
            color_pal = sns.color_palette("magma", len(u_vals))
            lut = dict(zip(u_vals, color_pal)) #Create a dictionary where the key is the category and the values are the colors from the palette we just created
            row_colors[label] = pd.Series(values).map(lut) #map the colors to the series. Now we have a list of colors the same length as our dataframe, where unique values are mapped to the same color
        row_colors = pd.DataFrame(row_colors)
        row_colors.index = CountMatrix_clinical[merge_column_inCount]

        display(CountMatrix_clinical.head())

        count_feat = remove_column(count_feat, merge_column_inCount)
        count_feat = remove_column(count_feat, merge_column_inClinical)
        for col in col_toRemove:
            count_feat = remove_column(count_feat, col)
        
        count_feat = count_feat.fillna(0)
        # row_colors.index = count_matrix.index

        print('MATRIX:')
        display(count_feat.head())

        # row_colors.index = count_matrix.index
        clustermap = sns.clustermap(count_feat, figsize=figsize, cmap=cmap, method = method)
        plt.show()

        clustermap = sns.clustermap(count_feat, figsize=figsize, row_colors = row_colors, cmap=cmap, method = method)
        plt.show()


    else:
        # Create the clustermap
        clustermap = sns.clustermap(
            count_matrix_filled,
            figsize=figsize,
            cmap=cmap,
            row_cluster=cluster_row,
            col_cluster=cluster_col,
            linewidths=linewidths, method = method
        )
        
    # Set plot title
    plt.title(title)

    # Export the figure if export_fig is specified
    if export_fig:
        plt.savefig(export_fig)
        print(f"Heatmap exported to: {export_fig}")

    # Show the plot
    plt.show()

# Plot a boxplot from a matrix where the x-axis corresponds to the columns.
def plot_boxplot(matrix, figsize=[20, 6], export_fig=False, fillna=0, cmap='Spectral', font_scale = 1, title = 'Heatmap', linewidths=.05):
    """
    - matrix (pd.DataFrame or np.ndarray): The input matrix (DataFrame or NumPy array).
    - figsize (tuple, optional): Size of the figure (width, height). Default is (10, 6).
    - xlabel (str, optional): Label for the x-axis. Default is 'Columns'.
    - ylabel (str, optional): Label for the y-axis. Default is 'Values'.
    - title (str, optional): Title of the plot. Default is 'Boxplot from Matrix'.
    """


    # If the input is a DataFrame, use Seaborn for plotting
    if isinstance(matrix, pd.DataFrame):
        # Melt the DataFrame to long format for Seaborn
        df_melted = matrix.melt(value_vars=matrix.columns, var_name='Columns', value_name='Values')

        # Set up the figure and axis
        plt.figure(figsize=figsize)

        # Create the boxplot using Seaborn with color by column
        sns.boxplot(x='Columns', y='Values', data=df_melted, palette=cmap)

        # Set labels and title
        plt.title(title)

        # Show the plot
        plt.show()
    else:
        print("Input matrix must be a DataFrame for Seaborn boxplot.")


def merge_count_clinical(count_matrix, clinical, merge_column_inClinical='Patient', merge_column_inCount='patient', extract_sample_pattern=False, extract_patient_pattern=False):
    # Extract sample names and patient names from the index
    if extract_sample_pattern:
        count_matrix['sample'] = count_matrix.index.to_series().str.extract(extract_sample_pattern)
    if extract_patient_pattern:
        count_matrix['patient'] = count_matrix.index.to_series().str.extract(extract_patient_pattern)

    if merge_column_inCount != merge_column_inClinical:
        clinical[merge_column_inCount] = clinical[merge_column_inClinical]
    # Merge count_matrix with clinical based on the specified merge_column
    df = count_matrix.merge(clinical, on=merge_column_inCount)
    return df

def can_be_numeric(series):
    try:
        pd.to_numeric(series)
        return True
    except (ValueError, TypeError):
        return False


import pandas as pd
from scipy.stats import pearsonr, spearmanr, kruskal, ttest_rel, ttest_ind
from scipy import stats

# Perform statistical tests between clinical and molecular data.
def perform_statistical_test(CountMatrix_clinical, clinical_columns, molecular_columns, columns_to_skip=['Patient','Sample']):
    """ Parameters:
    - CountMatrix_clinical (pd.DataFrame): The DataFrame containing clinical data.
    - clinical_columns (list): List of columns in CountMatrix_clinical containing clinical data.
    - molecular_columns (list): List of columns in CountMatrix_clinical containing molecular data.
    - columns_to_skip (list): List of columns to skip during the analysis (default=['Patient', 'Sample']).
    Returns:
    - results_df (pd.DataFrame): DataFrame containing the results of the statistical tests.
                                Columns: 'clinical_source', 'molecular_target', 'stat_value', 'pval', 'test_performed'. """
    results = []
    for clinical_col in clinical_columns:
        if clinical_col not in columns_to_skip:
            clinical_data = CountMatrix_clinical[clinical_col]

            for molecular_col in molecular_columns:
                skipped = 0
                if not molecular_col in columns_to_skip:
                    
                    molecular_data = CountMatrix_clinical[molecular_col]

                    # Determine if clinical data is numeric
                    clinical_data_type = 'Continuous Data' if can_be_numeric(clinical_data) else 'Categorical Data' 

                    # Exclude NaN values from the analysis
                    valid_indices = CountMatrix_clinical[clinical_data.notna() & molecular_data.notna()].index
                        
                    valid_indices = CountMatrix_clinical[clinical_data.notna() & molecular_data.notna()].index

                    
                    clinical_data_valid = clinical_data.loc[valid_indices]
                    molecular_data_valid = molecular_data.loc[valid_indices]
                    if len(clinical_data_valid.unique()) > 1:
                        # Perform appropriate statistical test
                        if clinical_data_type == 'Continuous Data':
                            try:
                                clinical_data_valid = pd.to_numeric(clinical_data_valid)
                            except:
                                pass
                            if clinical_data_valid.dtype in ['float64', 'int64']:
                                stat, pval = pearsonr(clinical_data_valid, molecular_data_valid)
                                test_result = f"pearsonr test between {clinical_col} and {molecular_col}"
                            else:  # If cell proportion column is not continuous
                                stat, pval = spearmanr(clinical_data_valid, molecular_data_valid)
                                test_result = f"spearmanr test between {clinical_col} and {molecular_col}"


                        elif clinical_data_type == 'Categorical Data':
                            if len(clinical_data_valid.unique()) < 20:
                                if len(clinical_data_valid.unique()) > 2:
                                    # Use one-way ANOVA for comparing means across multiple groups
                                    stat, pval = kruskal(*[molecular_data_valid[clinical_data == category] for category in clinical_data.unique()])
                                    test_result = f"Kruskal TEST between {clinical_col} and {molecular_col}"

                                    if pval > 0.05:
                                        stat, pval = stats.f_oneway(*[molecular_data_valid[clinical_data == category] for category in clinical_data.unique()])
                                        test_result = f"One-way ANOVA between {clinical_col} and {molecular_col}"


                                elif len(clinical_data_valid.unique()) == 2:
                                    unique_values = clinical_data_valid.unique()
                                    subset1 = molecular_data_valid[clinical_data_valid == unique_values[0]]
                                    subset2 = molecular_data_valid[clinical_data_valid == unique_values[1]]
                                    if len(subset1) == len(subset2):
                                        stat, pval = stats.ttest_rel(subset1, subset2)
                                        test_result = f"Paired t-test {clinical_col} and {molecular_col}"
                                    else:
                                        stat, pval = stats.ttest_ind(subset1, subset2)
                                        test_result = f"Independent t-test {clinical_col} and {molecular_col}"
                                else:
                                    skipped = 1
                            else:
                                skipped = 1
                    else:
                        skipped = 1
                    if not skipped:
                        # Append results to the table
                        results.append({
                            'clinical_source': clinical_col,
                            'molecular_target': molecular_col,
                            'stat_value': stat,
                            'pval': pval,
                            'test_performed': test_result
                        })

    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    return results_df

def generate_simulation_data(num_patients=100, seed=42):
    """
    Generate simulated clinical and molecular data for a given number of patients.

    Parameters:
    - num_patients (int): Number of patients to simulate (default=100).
    - seed (int): Seed for reproducibility (default=42).

    Returns:
    - simulated_data (pd.DataFrame): Simulated DataFrame containing clinical and molecular data.
    """
    np.random.seed(seed)

    # Simulate numeric clinical data
    numeric_data_columns = ['Age', 'BMI', 'Blood_Pressure', 'Cholesterol']
    numeric_data = pd.DataFrame({
        'Age': np.random.randint(18, 65, num_patients),
        'BMI': np.random.uniform(18.5, 30, num_patients),
        'Blood_Pressure': np.random.randint(90, 140, num_patients),
        'Cholesterol': np.random.randint(120, 240, num_patients),
    })

    # Simulate categorical clinical data
    categorical_data_columns = ['Sex', 'Smoking_Status', 'Diabetes', 'Hypertension']
    categorical_data = pd.DataFrame({
        'Sex': np.random.choice(['Male', 'Female'], num_patients),
        'Smoking_Status': np.random.choice(['Smoker', 'Non-Smoker', np.nan], num_patients),
        'Diabetes': np.random.choice(['Yes', 'No', np.nan], num_patients),
        'Hypertension': np.random.choice(['Yes', 'No', np.nan], num_patients),
    })

    # Concatenate numeric and categorical clinical data
    clinical_data = pd.concat([numeric_data, categorical_data], axis=1)

    # Simulate proportion of cells for each patient
    proportion_columns = [f'Cell_{i}' for i in range(1, 31)]
    proportion_data = pd.DataFrame(np.random.rand(num_patients, 30), columns=proportion_columns)

    # Concatenate clinical data and proportion data
    simulated_data = pd.concat([clinical_data, proportion_data], axis=1)

    return simulated_data


# Plot statistical results based on the output of perform_statistical_test function.
def plot_statistical_results(result_table, CountMatrix_clinical, threshold=0.05, boxplot_cmap = 'Spectral_r' , count_pos = 1, count_adj = -0.01,  count_color='blue', count_fontsize = 12, jitter_color='black', jitter_alpha = 0.5):
    """    Parameters:
    - result_table (pd.DataFrame): DataFrame containing statistical test results.
                                  Expected columns: 'clinical_source', 'molecular_target', 'stat_value', 'pval', 'test_performed'.
    - CountMatrix_clinical (pd.DataFrame): The original DataFrame containing clinical and molecular data.
    - threshold (float): Threshold for significance to determine whether to plot the results (default=0.05).
    - boxplot_cmap (str): Seaborn color map for the violin plot (default='Spectral_r').
    - count_pos (int): Vertical position for count annotations on the violin plot (default=1).
    - count_adj (float): Adjustment for count annotations on the violin plot (default=-0.01).
    - count_color (str): Color for count annotations (default='blue').
    - count_fontsize (int): Font size for count annotations (default=12).
    - jitter_color (str): Color for jitter points in the strip plot (default='black').
    - jitter_alpha (float): Alpha value for jitter points in the strip plot (default=0.5).
    """

    for index, row in result_table.iterrows():
        clinical_col = row['clinical_source']
        molecular_col = row['molecular_target']
        stat_value = row['stat_value']
        pval = row['pval']
        test_result = row['test_performed']

        # Check if the p-value is below the threshold for significance
        if pval < threshold:
            # Add violin plot on top
            if "ANOVA" in test_result or "t-test" in test_result or "Kruskal" in test_result:
                plt.figure(figsize=(10, 6))
                ax = sns.violinplot(x=clinical_col, y=molecular_col, data=CountMatrix_clinical, inner="quartile",
                                    palette=boxplot_cmap)
                sns.stripplot(x=clinical_col, y=molecular_col, data=CountMatrix_clinical, color=jitter_color,
                              jitter=True, dodge=True, alpha=jitter_alpha)

                # Add count annotations on top of each violin
                skip = 0
                for i, group in enumerate(CountMatrix_clinical[clinical_col].unique()):
                    count = CountMatrix_clinical[CountMatrix_clinical[clinical_col] == group][molecular_col].count()
                    if count > 0:
                        ax.text(i - skip, ax.get_ylim()[count_pos] + count_adj, f'n={count}', ha='center', va='bottom',
                            color=count_color, fontsize=count_fontsize)
                    else:
                        skip += 1

                plt.title(test_result + f"\n(p-value: {pval:.4f}, Stat Value: {stat_value:.4f})")
                plt.tight_layout()
                plt.show()
            else:
                # If not ANOVA, t-test, or Kruskal test, plot a regression plot
                plt.figure(figsize=(10, 6))
                sns.regplot(x=clinical_col, y=molecular_col, data=CountMatrix_clinical,
                            scatter_kws={'color': 'black', 'alpha': 0.5})
                plt.title(test_result + f"\n(p-value: {pval:.4f}, Stat Value: {stat_value:.4f})")
                plt.show()