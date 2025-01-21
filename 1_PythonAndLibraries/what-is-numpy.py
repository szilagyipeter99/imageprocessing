# Introduction to NumPy for Image Processing


import numpy as np

# --- Section 1: Creating NumPy Arrays ---
print("\n--- Section 1: Creating NumPy Arrays ---")

arr1d = np.array([1, 2, 3, 4, 5])
print("1D Array:", arr1d)
arr2d = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("2D Array (3x3):\n", arr2d)
zeros_array = np.zeros((2, 3))
ones_array = np.ones((3, 2))
print("Zeros array (2x3):\n", zeros_array)
print("Ones array (3x2):\n", ones_array)
range_array = np.arange(0, 10, 2)
print("Array with a range of numbers (0 to 10, step 2):", range_array)

input("Press Enter to continue...")
# --- End of Section 1 ---



# --- Section 2: Basic Array Operations ---
print("\n--- Section 2: Basic Array Operations ---")

a = np.array([5, 10, 15])
b = np.array([2, 3, 4])
print("a + b =", a + b)
print("a - b =", a - b)
print("a * b =", a * b)
print("a / b =", a / b)
print("a * 2 =", a * 2)
print("a + 5 =", a + 5)
print("Square root of a:", np.sqrt(a))

input("Press Enter to continue...")
# --- End of Section 2 ---



# --- Section 3: Indexing and Slicing ---
print("\n--- Section 3: Indexing and Slicing ---")

arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("First row:", arr2d[0, :])
print("Second column:", arr2d[:, 1]) 
print("First and last row:\n", arr2d[[0, -1]])

input("Press Enter to continue...")
# --- End of Section 3 ---



# --- Section 4: Reshaping and Resizing ---
print("\n--- Section 4: Reshaping and Resizing ---")

arr1d = np.arange(1, 10)
reshaped_arr = arr1d.reshape(3, 3)
print("1D array reshaped into 3x3:\n", reshaped_arr)
flattened_arr = reshaped_arr.flatten()
print("Flattened array:", flattened_arr)
resized_arr = np.resize(arr1d, (2, 5))  # Hover to check default
print("Resized array (2x5):\n", resized_arr)

input("Press Enter to continue...")
# --- End of Section 4 ---



# --- Section 5: Operations on entire arrays without loops ---
print("\n--- Section 5: Operations on entire arrays without loops ---")

arr = np.array([10, 20, 30])
print("Array + 5 (broadcasting):", arr + 5)
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
print("2D Array + scalar (broadcasting):\n", arr2d + scalar)
print("Square of all elements in arr2d:\n", np.square(arr2d))
print("Sum of all elements in arr2d:", np.sum(arr2d))

input("Press Enter to continue...")
# --- End of Section 5 ---



# --- Section 6: Basic Matrix Operations ---
print("\n--- Section 6: Basic Matrix Operations ---")

vec1 = np.array([1, 2, 3])
vec2 = np.array([4, 5, 6])
dot_product = np.dot(vec1, vec2)
print("Dot product of vec1 and vec2:", dot_product)
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])
print("Matrix 1:\n", matrix1)
print("Matrix 2:\n", matrix2)
print("Transpose of Matrix 1:\n", matrix1.T)
matmul_result = np.matmul(matrix1, matrix2)
print("Matrix multiplication (matmul):\n", matmul_result)
matmul_alt = matrix1 @ matrix2
print("Matrix multiplication (using @ operator):\n", matmul_alt)

input("Press Enter to continue...")
# --- End of Section 6 ---


print("\n--- End of NumPy Introduction ---")
