# --- STEP 3: VISUALIZATION (EXPLORATION) - REDUCED WINDOW ---

# Import libraries (Safety)
import matplotlib.pyplot as plt
import seaborn as sns

# *** CHANGE: DEFINE THE WINDOW TO PLOT ***
# Select the first 10,000 rows for a quick visualization that is light on RAM
WINDOW_SIZE = 10000
plot_window = df.head(WINDOW_SIZE)
print(f"Plotting performed on a reduced time window ({WINDOW_SIZE} rows).")

# Visualize sensor signals during an anomaly (Target = 1)
sensors_to_plot = ['sensor_04', 'sensor_07', 'sensor_50']

plt.figure(figsize=(14, 10))

for i, sensor in enumerate(sensors_to_plot):
    plt.subplot(len(sensors_to_plot), 1, i+1)

    # LINE PLOT: Use the reduced window
    plt.plot(plot_window.index, plot_window[sensor], label=sensor, color='blue', linewidth=1.0)

    # SCATTER PLOT: Use indices of anomalies within the reduced window
    anomaly_indices_window = plot_window[plot_window['target'] == 1].index

    if not anomaly_indices_window.empty:
        plt.scatter(
            anomaly_indices_window,
            plot_window.loc[anomaly_indices_window, sensor],
            color='red',
            label='ANOMALY (Target=1)',
            s=10,
            alpha=0.7
        )

    plt.legend()
    plt.title(f'Evolution of {sensor} with Anomalies (Red Points) - Reduced Time Window')

plt.tight_layout()
plt.show()

# Correlation Matrix
# This does not use long time series plots and should always work
cols_subset = ['sensor_04', 'sensor_07', 'sensor_10', 'sensor_17', 'sensor_42', 'target']
plt.figure(figsize=(8, 7))
sns.heatmap(df[cols_subset].corr(), annot=True, fmt="_]()
