import cv2 as cv
import numpy as np

# Image source: Case courtesy of Tariq Walizai, Radiopaedia.org, rID: 184833
# https://radiopaedia.org/cases/184833?lang=us
cv_image = cv.imread("segmentation/chest_tube.jpg", 0)
assert cv_image is not None, "File could not be read"

data = np.asarray(cv_image)
print(data.shape) # (height, width, channels)

seed_points = [[135,235],
                [360,215],
                [115,335]] # (X,Y) coordinates of points

color_values = [(255, 0, 0),
                (0, 255, 0),
                (0, 0, 255)] # (B, G, R)

# Mark the seed points with a red dot
color_image = cv.cvtColor(cv_image,cv.COLOR_GRAY2RGB)
for p in seed_points: 
    color_image[p[1] - 5 : p[1] + 5, p[0] - 5 : p[0] + 5] = (0, 0, 255) # (B, G, R)
# Empty matrix for tracking visited points
# Double parentheses (())
visited_points = np.zeros((data.shape[1], data.shape[0]), dtype=np.uint8)

temp_image = cv.cvtColor(cv_image,cv.COLOR_GRAY2RGB)
tolerance = 2 # Percent difference

for p in range(len(seed_points)):
    temp_list = [seed_points[p]]
    while (len(temp_list) != 0):
        curr_point = temp_list.pop() # Read then remove the last element of the list
        x = curr_point[0]
        y = curr_point[1]
        im_val = data[y][x] # Pixel value at the current point
        visited_points[y][x] = 1 # Mark current point white
        temp_image[y][x] = color_values[p]
        # Visit every neighbor (8-connectivity)
        neighbors_8c = [(x-1, y-1), (x-1, y+1), (x+1, y-1), (x+1, y+1), (x, y-1), (x, y+1), (x-1, y), (x+1, y)]
        for i, j in neighbors_8c:
            if (0 <= j < data.shape[0] and 0 <= i < data.shape[1]): # The X and Y index are inside the image
                neighbor_val = data[j][i]
                # Change the data type for proper subtraction
                neighbor_val = neighbor_val.astype(np.int16)
                # The point has not been visited and is similar to its neighbor
                if (visited_points[j][i] == 0 and abs(neighbor_val - im_val) < tolerance / 100 * 255 ):
                    # LIFO method
                    temp_list.append([i,j])

segmented_image = temp_image

cv.imshow("Window title", segmented_image)
k = cv.waitKey(0) # Wait for a keystroke in the windows     
