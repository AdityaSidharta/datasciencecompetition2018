import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(csv_path):
    data = pd.read_csv(csv_path)
    average = np.sum(np.arange(34) * data.values[:,2:], axis=1) /2016
    data['ave'] = average
    sns.set()
    viz = data.pivot("lat", "long", "ave")
    f, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(csv_path)
    sns.heatmap(viz, square=True)


