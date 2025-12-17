# Introduction to Python for Image Processing


# --- Section 1: Mathematical Operations ---
print("\n--- Section 1: Mathematical Operations ---")

a = 5
b = 2
c = "Python is the best!"
sum_value = a + b
diff_value = a - b
prod_value = a * b
div_value = a / b
print("a + b = ", sum_value)
print("a - b = ", diff_value)
print("a * b = ", prod_value)
print("a / b = ", div_value)
print("a ^ b = ", a ** b)
print(c)

input("Press Enter to continue...")
# --- End of Section 1 ---



# --- Section 2: If-Else Statements ---
print("\n--- Section 2: Control Flow: If-Else Statements ---")

threshold = 100
pixel_value = 120
if pixel_value > threshold:
    print("Pixel is brighter than the threshold.")
else:
    print("Pixel is darker than the threshold.")

input("Press Enter to continue...")
# --- End of Section 2 ---



# --- Section 3: Loops: For and While ---
print("\n--- Section 3: Loops: For and While ---")

for i in range(5):
    print("For loop iteration:", i)
count = 0
while count < 5:
    print("While loop count:", count)
    count += 1
rows, cols = 2, 4
for row in range(rows):
    for col in range(cols):
        print(f"Processing pixel at ({row}, {col})")

input("Press Enter to continue...")
# --- End of Section 3 ---



# --- Section 4: Lists and Indexing ---
print("\n--- Section 4: Lists and Indexing ---")

pixel_values = [34, 67, 89, 120, 255, 0]
print("First pixel value:", pixel_values[0])
print("Last pixel value:", pixel_values[-1])

input("Press Enter to continue...")
# --- End of Section 4 ---



# --- Section 5: Advanced Indexing ---
print("\n--- Section 5: Advanced Indexing ---")

pixel_values = [34, 67, 89, 120, 255, 0, 150, 200, 175, 90]
print("First 3 elements:", pixel_values[0:3])
print("Last 3 elements:", pixel_values[-3:])
print("Middle 4 elements:", pixel_values[3:7])
print("Every second element:", pixel_values[::2])
print("Reversed values:", pixel_values[::-1])
image_grid = [
    [34, 67, 89],  # Row 1
    [120, 255, 0],  # Row 2
    [150, 200, 175]  # Row 3
]
print("Pixel at (0, 0):", image_grid[0][0])  # First row, first column
print("Pixel at (2, 1):", image_grid[2][1])  # Third row, second column
print("First row:", image_grid[0])
print("Last row:", image_grid[-1])
print("All pixel values in the grid:")
for row in image_grid:
    for pixel in row:
        print(pixel, end = " ")
    print()

input("Press Enter to continue...")
# --- End of Section 5 ---



# --- Section 6: Functions ---
print("\n--- Section 6: Functions ---")

def calculate_average(a, b):
    return (a + b) / 2
average_value = calculate_average(100, 150)
print("The average value is:", average_value)

input("Press Enter to continue...")
# --- End of Section 6 ---



# --- Section 7: Dictionaries ---
print("\n--- Section 7: Dictionaries ---")

image_metadata = {
    "filename": "image1.jpg",
    "width": 1920,
    "height": 1080,
    "color_mode": "RGB"
}
print("Image filename:", image_metadata["filename"])
image_metadata["filename"] = "image2.png"
print("Updated filename:", image_metadata["filename"])

input("Press Enter to continue...")
# --- End of Section 7 ---



# --- Section 8: Exception Handling ---
print("\n--- Section 8: Exception Handling ---")

try:
    result = 10 / 0
except ZeroDivisionError:
    print("Error: Division by zero is not allowed.")
finally:
    print("This runs no matter what!")

input("Press Enter to continue...")
# --- End of Section 8 ---


print("\n--- End of Python Introduction ---")
