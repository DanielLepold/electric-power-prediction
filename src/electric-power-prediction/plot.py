import functools
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import logging

def create_histograms(df,path):
  folder_path=path+"/Plots/Histograms"
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  # Iterate over each column
  for column in df.columns:
    # Plot histogram for the current column
    df[column].hist(bins=30)

    # Add labels and title
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Histogram of {column}')

    plt.savefig(os.path.join(folder_path,f'histogram_{column}.png'))
    # Show the plot
    plt.show()

def create_density_plots(df, path):
  folder_path=path+"/Plots/Density"
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  # Iterate over each column
  for column in df.columns:
    # Plot histogram for the current column
    df[column].plot.kde()

    # Add labels and title
    plt.xlabel(column)
    plt.ylabel('Density')
    plt.title(f'Density of {column}')

    plt.savefig(os.path.join(folder_path,f'density_{column}.png'))
    # Show the plot
    plt.show()

def create_scatter_plots(df, path):
  folder_path = path+"/Plots/Scatter"
    # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

    # Iterate over each column
  for column in df.columns:
    # Extract the column data
    column_data = df[column]

    # Sort the data
    sorted_data = column_data.sort_values()

    # Calculate probabilities
    p = np.arange(0, len(sorted_data)) / len(sorted_data)

    # Create scatter plot
    plt.scatter(sorted_data, p, s=5)

    # Add labels and title
    plt.xlabel(column)
    plt.ylabel('Probability')
    plt.title(f'Scatter Plot of {column}')

    plt.savefig(os.path.join(folder_path,f'scatter_{column}.png'))

    # Show the plot
    plt.show()

def create_box_plots(df,path):
  folder_path=path+"/Plots/Box"
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  for column in df.columns:
    # Create a boxplot for the current column
    df[[column]].boxplot(showfliers=True, flierprops=dict(marker='o', markersize=1))

    # Add title
    plt.title(f'Boxplot of {column}')

    plt.savefig(os.path.join(folder_path,f'box_{column}.png'))

    # Show the plot
    plt.show()

def create_violin_plots(df, path):
    folder_path=path+"/Plots/Violin"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    for column in df.columns:
      # Create a violin plot for the current column
      plt.figure(figsize=(8, 6))  # Optional: adjust figure size if needed
      plt.violinplot(df[column], showmeans=False, showmedians=True)

      # Add title
      plt.title(f'Violin Plot of {column}')

      # Save the plot
      plt.savefig(os.path.join(folder_path,f'violin_{column}.png'))

      # Show the plot
      plt.show()

def create_qq_plots(df,path):
  folder_path=path+"/Plots/QQ"
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  for column in df.columns:
    # Standardize the column
    column_std = (df[column] - df[column].mean()) / df[column].std()

    # Create a QQ plot for the standardized column
    stats.probplot(column_std, dist="norm", plot=plt)

    # Add title
    plt.title(f'QQ Plot of {column}')

    # Save the plot
    plt.savefig(os.path.join(folder_path,f'qq_{column}.png'))

    # Show the plot
    plt.show()

def create_all_pairplots(df,path):
    folder_path=path+"/Plots/Pair"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)

    # Get all combinations of columns
    columns = df.columns
    combinations = [(x, y) for x in columns for y in columns]

    # Plot pairplot for each variation
    for x, y in combinations:
      # Create pairplot
      pairplot = sns.pairplot(df, x_vars=x, y_vars=y,plot_kws={'s': 2})

      # Save pairplot to the specified folder
      pairplot.savefig(os.path.join(folder_path, f'pairplot_{x}_{y}.png'))

      # Close the pairplot figure to release memory
      plt.close(pairplot.fig)

def create_jointplots(df, path):
  folder_path=path+"/Plots/Joint"
  # Create the folder if it doesn't exist
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

  # Plot jointplot for each combination
  for x in df.columns:
    if x == "PE":
      continue;
    # Create jointplot
    jointplot = sns.jointplot(data=df, x=x, y="PE")

    # Save jointplot to the specified folder
    jointplot.savefig(os.path.join(folder_path, f'jointplot_{x}_PE.png'))

    # Close the jointplot figure to release memory
    plt.close(jointplot.fig)

def create_heatmap(corr_matrix,path):
    folder_path=path+"/Plots/Heatmap"
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
      os.makedirs(folder_path)
    # Create heatmap
    sns.heatmap(corr_matrix, annot=True)

    # Save the plot
    plt.savefig(os.path.join(folder_path,"heatmap.png"))

    # Show the plot
    plt.show()

def create_plots(df,folder_path):
    logging.info("Plot creation started.")
    # Define the plotting functions
    plotting_functions = [
      create_scatter_plots,
      create_density_plots,
      create_histograms,
      create_box_plots,
      create_violin_plots,
      create_qq_plots,
      create_all_pairplots,
      create_jointplots
    ]

    # Call each plotting function with the DataFrame and folder path
    list(map(functools.partial(lambda f, df, folder_path: f(df, folder_path),
                               folder_path=folder_path, df=df),
             plotting_functions))

    # Creating heat map
    corr = df.corr()
    create_heatmap(corr,folder_path)
    logging.info("Plot creation finished.")
