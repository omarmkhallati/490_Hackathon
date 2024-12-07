import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load The Survey Data
data = pd.read_csv ("C:/Users/user/Desktop/2024_PersonalityTraits_SurveyData.csv")

# Select The Features Of Interest
features = [
    "Do you find it difficult to refrain from smoking where it is forbidden (church, library, cinema, plane, etc...)?",
    "How many cigarettes do you smoke each day?",
    "Do you smoke more frequently during the first hours after waking up than during the rest of the day? ",
    "Do you smoke if you are so ill that you are in bed most of the day?",
    "How soon after you wake up do you smoke your first cigarette?",
    "Which cigarette would you mostly hate to give up?",
    "How old were you the first time you smoked a full cigarette (not just a few puffs)?",
    "How would you describe your current smoking behavior compared to your smoking behavior before Lebanon's economic crisis and revolution began in 2019?"
]

# Keep Only The Relevant Features
data = data [features]

# Encode Categorical Data Into Numbers
label_encoders = {}
for col in data.columns :
    if data [col].dtype == 'object' :
        le = LabelEncoder()
        data [col] = le.fit_transform(data[col].fillna('Unknown'))
        label_encoders [col] = le

# Convert All Data To Numeric And Remove Missing Values
data = data.apply (pd.to_numeric, errors = 'coerce').dropna()

# Convert Data Into A NumPy Array For Processing
data_np = data.to_numpy ()

# Perform PCA To Reduce Dimensionality To 2D
pca = PCA (n_components = 2)
data_2d = pca.fit_transform (data_np)

# Function To Remove Outliers Based On IQR
def remove_outliers (data, threshold = 1.5):
    Q1 = np.percentile (data, 25, axis = 0)
    Q3 = np.percentile (data, 75, axis = 0)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    mask = np.all ((data >= lower_bound) & (data <= upper_bound), axis = 1)
    return data [mask] , mask

# Remove Outliers From The PCA Data
data_2d_clean, mask = remove_outliers (data_2d)

# Filter Original Data To Only Include Non-Outlier Rows
data_original = data [mask]

# Initialize K-Means Centroids
centroids = np.array ([[2, 3], [2, -1]])

max_iterations = 100  # Set The Number Of Iterations
convergence_threshold = 1e-4  # Convergence Condition
iteration = 0
ms = 10  # Marker Size For Plotting

prev_centroids = np.random.rand (2, 2)  # Set Initial Centroids

# Initial Plot Before Running K-Means
fig, ax = plt.subplots ()
clusters = np.zeros (len(data_2d_clean))
for i, point in enumerate (data_2d_clean) :
    ax.plot (point[0], point[1], '.', color = 'C0' if clusters[i] == 0 else 'C1', ms = ms)
ax.plot (centroids[:, 0], centroids[:, 1], 'X', color = 'C2', markersize = ms)
plt.show ()

# Run The K-Means Clustering Loop
while np.sum (np.abs(prev_centroids - centroids)) > convergence_threshold and iteration < max_iterations :
    print (f"Iteration {iteration}")
    prev_centroids = centroids.copy()
    clusters = np.zeros (len(data_2d_clean))
    
    # Assign Points To The Nearest Centroid
    for i, point in enumerate (data_2d_clean) :
        distances = np.sum ((point - centroids)**2, axis=1)
        clusters [i] = np.argmin(distances)

    # Update Centroids Based On Cluster Assignments
    for j in range (2):
        if np.any (clusters == j): 
            centroids[j] = np.mean (data_2d_clean[clusters == j], axis = 0)
    
    iteration += 1

    # Plot The Data With Updated Centroids At Each Iteration
    fig, ax = plt.subplots()
    for i, point in enumerate (data_2d_clean):
        ax.plot (point[0], point[1], '.', color = 'C0' if clusters[i] == 0 else 'C1', ms = ms)
    ax.plot (centroids[:, 0], centroids[:, 1], 'x', color = 'C2', markersize = ms)
    ax.set_title (f'Iteration {iteration}')
    plt.show()

# Assign Clusters To The Original Data And Save Results
data_original['Cluster'] = clusters

# Save The Clustered Data To A CSV File
data_original.to_csv("C:/Users/user/Desktop/K-Means_Output.csv", index=False)