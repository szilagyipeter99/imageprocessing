import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Open the image using PIL
image = Image.open("resources/wrench.png")

# Define input and output points (top-left, top-right, bottom-left, bottom-right)
input_points = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
output_points = np.float32([[10, 100], [180, 20], [30, 250], [200, 220]])

def compute_perspective_matrix(inp_pts, out_pts):
    A = []
    b = []
    # Populate the two lists (arrays) with values
    for i in range(4):
        x, y = inp_pts[i][0], inp_pts[i][1]
        u, v = out_pts[i][0], out_pts[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.append(u)
        b.append(v)
    # Solve the equation system
    h = np.linalg.solve(A, b)
    # Add 1 as the last element and reshape into a 3x3 matrix
    # (Reshape is unneccessary in this exact application, because it is flattened in the next step)
    H = np.append(h, 1).reshape(3, 3)
    return H

perspective_matrix = compute_perspective_matrix(input_points, output_points)

# PIL expects a flattened 8-element matrix
# Flatten the matrix and extract the first 8 elements
matrix_for_pil = perspective_matrix.flatten()[:8]

# Apply the transform to the original image
transformed_image = image.transform(image.size, Image.Transform.PERSPECTIVE, matrix_for_pil, resample = Image.Resampling.BICUBIC)

# Display images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].imshow(image)
ax[0].set_title("Original Image")
ax[0].axis("off")
ax[1].imshow(transformed_image)
ax[1].set_title("Transformed Image")
ax[1].axis("off")
plt.show()