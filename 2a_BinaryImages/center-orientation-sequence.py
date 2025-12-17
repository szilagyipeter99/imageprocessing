from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Folder containing the image sequence
folder_path = "path-to-resources/wrench-sequence"

# Set threshold value
threshold = 175

# Loop through each image in the folder
for filename in sorted(os.listdir(folder_path)):
    if filename.startswith("wrench00108"):
        # Open the image and convert to grayscale
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path).convert("L")

        # Convert image to a NumPy array
        data = np.array(image, dtype = np.uint8)

        # Apply thresholding
        # Convert image from [0, 255] to {0, 1}
        data = np.where(data > threshold, 1, 0)

        # Helper arrays to calculate the center
        x_range = np.arange(0, data.shape[1])
        y_range = np.arange(0, data.shape[0])

        # Calculate area and center
        area = data.sum()
        x_cntr = np.matmul(data, x_range).sum() / area
        y_cntr = np.matmul(data.T, y_range).sum() / area

        # --- PCA (Principal Component Analysis) --- 
        # Extract foreground pixel coordinates
        y, x = np.nonzero(data)
        coords = np.column_stack((x, y))
        # Centered array
        cntr_coords = coords - (x_cntr, y_cntr)
        # Covariance matrix
        cov_matrix = np.cov(cntr_coords, rowvar = False)
        # Eigen value decomposition (EVD) to find the principal components
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        # Eigenvector corresponding to the largest eigenvalue
        pr_eig_vec = eig_vecs[:, np.argmax(eig_vals)]
        # Orientation angle in radians
        orientation = np.arctan2(pr_eig_vec[1], pr_eig_vec[0])

        # Start and end point for the orientation line
        half_len = 300
        x_line = [x_cntr - half_len * np.cos(orientation), x_cntr + half_len * np.cos(orientation)]
        y_line = [y_cntr - half_len * np.sin(orientation), y_cntr + half_len * np.sin(orientation)]

        # Display the image
        plt.figure(1); plt.clf() # This is needed to refresh the image
        plt.imshow(image, cmap = "gray")
        plt.plot(x_line, y_line, color = "red", linewidth = 3)  # Draw a red line on the image
        plt.plot(x_cntr, y_cntr, "og", markersize = 5)  # Mark center with a green dot
        plt.title(f"Orientation of {filename}")
        plt.axis('off')
        plt.pause(.033)  # This is needed to refresh the image
