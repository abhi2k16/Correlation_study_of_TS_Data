</header>
The script is a Python program designed for analyzing time-series data, particularly focusing on correlations between various parameters and N2_RATE. It includes the following key functionalities:

# Data Import and Cleaning:

  Reads a CSV file containing time-series data.
  Cleans the data by filling NaN values and removing unwanted data (e.g., negative values in specific columns like PTEMP_RAW and IPRS_RAW).
Visualization:

Plots individual parameters over time for exploratory analysis.
Implements multi-axis plots to compare multiple parameters simultaneously.
Correlation Analysis:

  Defines a correlation class with methods to calculate and visualize correlations:
  scatter_plot: Plots scatter plots of normalized data.
  corr_plot: Calculates and plots Pearson correlation coefficients over time.
  corr_param: Calculates Kendall Tau correlations for different time steps.
  corr_slid: Calculates and visualizes sliding window correlations using Spearman correlation.
  corr_slid_timeLag: Analyzes correlations with time lags.
Noise Quantification:

  Implements methods to quantify noise in data using techniques like filtering, correlation coefficients, Jensen-Shannon       divergence, and coefficient of variation.
Outlier Analysis:

Plots specific parameters against IPRS_RAW to identify potential outliers.
Overall Correlation Measurement:

  Computes and visualizes a correlation matrix heatmap for selected parameters using Seaborn.
  The script is modular, with reusable methods for correlation analysis and visualization, making it suitable for exploratory data analysis and event detection in time-series datasets.
