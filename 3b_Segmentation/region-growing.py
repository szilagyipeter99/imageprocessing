from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Open the image using PIL
# Image source: Case courtesy of Tariq Walizai, Radiopaedia.org, rID: 184833
# https://radiopaedia.org/cases/184833?lang=us
image = Image.open("path-to-resources/chest-tube.jpg").convert("RGB")

# Convert image to a NumPy array
data = np.array(image, dtype=np.uint8)
# Create a grayscale version, as well
grayscale_data = np.array(image.convert("L"), dtype=np.uint8)

seed_points = [[135,235],
               [360,215],
               [115,335]] # (X,Y) coordinates of points

color_values = [(255, 0, 0),
                (0, 255, 0),
                (0, 0, 255)] # (R, G, B)

# Create empty matrix to track visited points
visited_points = np.zeros((data.shape[1], data.shape[0]), dtype=np.uint8)

segmented_image = data.copy()

# Max. difference between pixels (%)
tolerance = 2

for p in range(len(seed_points)):
    temp_list = [seed_points[p]]
    while (len(temp_list) != 0):
        curr_point = temp_list.pop() # Read then remove the last element of the list
        x = curr_point[0]
        y = curr_point[1]
        im_val = grayscale_data[y][x] # Pixel value at the current point
        visited_points[y][x] = 1 # Mark current point white
        segmented_image[y][x] = color_values[p]
        # Visit every neighbor (8c)
        neighbors = [(x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1), (x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        for i, j in neighbors:
            if (0 <= j < data.shape[0] and 0 <= i < data.shape[1]): # The X and Y index are inside the image
                neighbor_val = grayscale_data[j][i]
                # Change the data type for proper subtraction
                neighbor_val = neighbor_val.astype(np.int16)
                # The point has not been visited and is similar to its neighbor
                if (visited_points[j][i] == 0 and abs(neighbor_val - im_val) < tolerance / 100 * 255 ):
                    # LIFO method
                    temp_list.append([i,j])

# Display the image
plt.imshow(segmented_image)
# Mark seed points
plt.plot(seed_points[0][0], seed_points[0][1], "ow", markersize=5)
plt.plot(seed_points[1][0], seed_points[1][1], "ow", markersize=5)
plt.plot(seed_points[2][0], seed_points[2][1], "ow", markersize=5)
plt.title("Center point")
plt.axis('off')
plt.show()   
