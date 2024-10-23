import numpy as np
import os

# Load the coordinates to inspect
coordinates_file = os.path.join('output', f"ball_positions.npy")

xy_coordinates = np.load( coordinates_file, allow_pickle=True)
print("Loaded coordinates:", xy_coordinates)
print("Shape of loaded coordinates:", xy_coordinates.shape)
