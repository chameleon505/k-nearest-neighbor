# Importing necessary libraries
import numpy as np
import pandas as pd
import math
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import statistics
from statistics import mode

# Function to compute Euclidean distance between two points
def euclidian_distance(point_1, point_2):
    sum = 0
    sum = (point_1['x-coordinate'] - point_2['x-coordinate']) ** 2 + (point_1['y-coordinate'] - point_2['y-coordinate']) ** 2
    return math.sqrt(sum)

# Reading data from CSV file and preprocessing
df = pd.read_csv('Cluster_Data_5.csv')
df.rename(columns={df.columns[0]: "A"}, inplace=True)
df[['x-coordinate', 'y-coordinate', 'class']] = df['A'].str.split(' ', expand=True)
df.drop('A', axis=1, inplace=True)
df['x-coordinate'] = pd.to_numeric(df['x-coordinate'])
df['y-coordinate'] = pd.to_numeric(df['y-coordinate'])
df['class'] = pd.to_numeric(df['class'])

# Dictionary mapping class labels to colors
color_dict = {1: ["blue", [31, 119, 180]],
              2: ["orange", [255, 127, 14]],
              3: ["green", [44, 160, 44]],
              4: ["red", [214, 39, 40]],
              5: ["magenta", [148, 103, 189]]
              }
classes = df['class'].tolist()

cols = []
for c in classes:
    cols.append(color_dict[c][0])

# Function to plot the original data points
def plot_cluster(data):
    plt.figure(figsize=(15, 12))
    x = data['x-coordinate']
    y = data['y-coordinate']
    plt.scatter(x, y, c=cols)
    tick_list = np.arange(-13, 13, 2.5)

    plt.xticks(tick_list, tick_list)

    plt.show()


# Function to find the k nearest neighbors of a new data point
def k_nearest_neighbour(new_point, data, k):
    distances = []
    points_number = len(data.index)

    for i in range(0,points_number) :
        distance = euclidian_distance(new_point, df.iloc[i])
        distances.append((df.iloc[i], distance))
    distances.sort(key=lambda x: x[1])
    return distances[:k]

# Function to determine the class label for a new data point
def define_class(new_point):

    nearest_neighbors = k_nearest_neighbour(new_point, df, 10)

    points = [point[0] for point in nearest_neighbors]
    top_k_classes = [point['class'] for point in points]

    return mode(top_k_classes)

# Define grid boundaries and create a meshgrid for visualization
x_min, x_max = -13, 13
y_min, y_max = -13, 13
grid_size = 1.0

xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_size), np.arange(y_min, y_max, grid_size))
points = np.c_[xx.ravel(), yy.ravel()]

# Calculate middle points of grid squares
middle_points = np.stack([(xx[:-1, :-1] + xx[1:, 1:]) / 2, (yy[:-1, :-1] + yy[1:, 1:]) / 2], axis=-1)

middle_points = middle_points.reshape(-1, 2)

list_points = []

for element in middle_points:
    new_point = {'x-coordinate': element[0], 'y-coordinate': element[1]}
    list_points.append(new_point)

list_classes = []
for m in list_points:
    new_class = define_class(m)
    list_classes.append(new_class)

# Create a scatter plot with colored grid squares representing class boundaries
fig = plt.figure(figsize=(15, 12))
ax = fig.add_subplot(111)

x = df['x-coordinate']
y = df['y-coordinate']
colors = []
for c in list_classes:
    colors.append(color_dict[c][0])
ax.scatter(x, y, c=cols)

for point, point_class in zip(middle_points, list_classes):
    rect = Rectangle(point - [grid_size / 2, grid_size / 2], grid_size, grid_size,
                     facecolor=color_dict[point_class][0], alpha=0.3)

    ax.add_patch(rect)

tick_list = np.arange(-13, 13, 1.0)

plt.xticks(tick_list, tick_list)
plt.yticks(tick_list, tick_list)

plt.grid(linewidth=0.5)

plt.show()

