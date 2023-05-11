def load():
    import pandas as pd
    print('Loaded')


def get_abudence(df, column_name):
    column = df[column_name]
    total_count = len(column)
    counts = column.value_counts()
    percentages = counts.apply(lambda x: (x / total_count) * 100)
    return percentages



def QC_cell_count(coords_files, figs_path):
    import matplotlib.pyplot as plt
    import secrets
    import string

    # Group data by filename and count number of rows in each group
    counts = coords_files.groupby('filename').size()

    # Create histogram of counts
    # counts.distplot(kind='hist', bins= 100)
    import seaborn as sns

    # Determine approximate bin width
    binwidth = (counts.max() - counts.min()) / len(counts)

    sns.histplot(data=counts, binwidth=binwidth, kde=True)


    # Set plot title and axis labels
    plt.title('Histogram of Number of cell per File')
    plt.xlabel('Number of cells')
    plt.ylabel('Frequency')

    # Generate a random filename
    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    outputname = f"histogram_{random_string}.png"
    plt.savefig(figs_path / outputname)

    # Show the plot
    plt.show()


def hist_compare_ROIs(coords_files, figs_path, id_col ='filename', ROIs_colname = 'ROIs', colors= ['blue', 'red'] , xlab = "Number of cells"):
    import matplotlib.pyplot as plt
    import secrets
    import string
    import seaborn as sns
    cell_count_by_ROIs = coords_files[[id_col,ROIs_colname]].value_counts().reset_index()
    g = sns.histplot(data=cell_count_by_ROIs, x=0, hue= ROIs_colname, kde=True, bins=10, multiple='stack', palette=colors, alpha=0.05)
    g.set_xlabel(xlab)
    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    outputname = f"histogramROIs_{random_string}.png"
    plt.savefig(figs_path / outputname)
    plt.show()


def hist_phenotype(coords_files, figs_path, id_col ='filename', ROIs_colname = 'ROIs', phenotype_col = 'phenotype'):
    import matplotlib.pyplot as plt
    import secrets
    import string
    import seaborn as sns
    cell_count_by_ROIs = coords_files[['filename','phenotype','ROIs']].value_counts().reset_index()

    sns.boxplot(x='phenotype', y=0, data=cell_count_by_ROIs,  hue='ROIs', palette='Spectral')
    plt.show()
    sns.boxplot(x='ROIs', y=0, data=cell_count_by_ROIs,  hue='phenotype', palette='Spectral')
    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    outputname = f"histogramROIs_{random_string}.png"
    plt.savefig(figs_path / outputname)
    plt.show()

    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    outputname = f"histogramROIs_{random_string}.png"
    plt.savefig(figs_path / outputname)
    random_string = ''.join(secrets.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    outputname = f"histogramROIs_{random_string}.png"
    plt.savefig(figs_path / outputname)
    plt.show()