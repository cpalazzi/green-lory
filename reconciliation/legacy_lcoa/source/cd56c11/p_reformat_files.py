import os
import re
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def rename_files():
    directory = os.getcwd() + r'\Hourly QLD Results'  # Replace with your directory path
    unwanted_text = 'Fix.xlsx'  # Replace with the text you want to remove

    # Get the list of files in the directory
    file_list = os.listdir(directory)

    # Iterate over each file
    for filename in file_list:
        # Check if the file is an Excel spreadsheet
        if filename.endswith('.xlsx'):
            # Remove the unwanted text using regular expressions
            new_filename = re.sub(unwanted_text, '.xlsx', filename)

            # Construct the full path for the original and new file
            original_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(original_path, new_path)

            print(f"Renamed '{filename}' to '{new_filename}'")


def aggregate_annual_results():
    file_lst = glob.glob(os.getcwd() + r'/QLD Summary Results/*.csv')
    df = None
    for file in file_lst:
        if df is None:
            df = pd.read_csv(file).rename(columns={'Unnamed: 0': 'Location'}).set_index('Location')#
            df.drop('0_0', inplace=True)
            scenario = [file.split('\\')[-1][7:-4] for _ in range(len(df))]
            year = [file.split('\\')[-1][:4] for _ in range(len(df))]
            df.insert(0, 'Year', year)
            df.insert(0, 'Scenario', scenario)
        else:
            df1 = pd.read_csv(file).rename(columns={'Unnamed: 0': 'Location'}).set_index('Location')
            df1.drop('0_0', inplace=True)
            scenario = [file.split('\\')[-1][7:-4] for _ in range(len(df1))]
            year = [file.split('\\')[-1][:4] for _ in range(len(df1))]
            df1.insert(0, 'Year', year)
            df1.insert(0, 'Scenario', scenario)
            df = pd.concat([df, df1])
    df.to_csv('All_in_summary.csv')


def plot_results():
    df = pd.read_csv('All_in_summary.csv')
    location = 'Banana_rez'

    df = df.loc[df.Location == location]
    df.Objective *= 1000

    sns.set_theme()
    x = np.arange(3)

    y1 = df.loc[df.Scenario == 'Ammonia_Flex'].Objective.to_list()
    y2 = df.loc[df.Scenario == 'Ammonia_Fix'].Objective.to_list()
    y3 = df.loc[df.Scenario == 'Ammonia_Fix_H'].Objective.to_list()
    width = 0.25

    # plot data in grouped manner of bar type
    plt.bar(x - 0.25, y1, width)
    plt.bar(x, y2, width)
    plt.bar(x + 0.25, y3, width)

    plt.xticks(x, ['2030', '2040', '2050'])
    plt.title('{a} Renewable energy zone'.format(a=location.split('_')[0]))

    plt.show()

if __name__ == '__main__':
    plot_results()
    # rename_files()