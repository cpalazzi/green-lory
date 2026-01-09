import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

def plot_monthly_results(file):
    df = pd.read_excel(file, sheet_name='Energy consumption')
    pass


if __name__ == '__main__':
    plot_monthly_results(r'Results\Banana_rez_2030 - Ammonia_operating.xlsx')