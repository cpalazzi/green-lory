import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter, LogLocator

def plot_storage_comparison():
    """Plots different storage technologies against each other"""

    # Data for Australia in AUD/MWh
    cycles_per_year = {'BESS': 720,
                       'PHES': 365,
                       'CHGS': 365,
                       'Salt\nCavern': 12,
                       'ammonia': 8760/2}
    CAPEX = {'BESS': 394000,
                       'PHES': 17142,
                       'CHGS': 36258,
                       'Salt\nCavern': 362,
                       'ammonia': 1040}
    y_adj = {'BESS': 150,
                       'PHES': 100,
                       'CHGS': -150,
                       'Salt\nCavern': -2,
                       'ammonia': -1500}
    x_adj = {'BESS': 0,
                       'PHES': 0,
                       'CHGS': 0,
                       'Salt\nCavern': 300,
                       'ammonia': 0}

    sns.set_style(style='darkgrid')
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    xs = [CAPEX[key] for key in CAPEX.keys()]
    ys = [cycles_per_year[key] for key in CAPEX.keys()]
    ax.scatter(xs, ys)

    # Set logarithmic scales on both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Storage CAPEX (AUD/MWh)')
    ax.set_ylabel('Cycles/year')
    ax.set_ylim([5, 1E4])
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    for key in CAPEX.keys():
        ax.annotate(text=key,
                xy=(CAPEX[key]+ x_adj[key], cycles_per_year[key] + y_adj[key]), ha='center', fontsize=10)

    # plt.show()
    fig.savefig('Storage_Cycling costs.png', bbox_inches='tight', format='png', dpi=1000)


if __name__ == '__main__':
    plot_storage_comparison()