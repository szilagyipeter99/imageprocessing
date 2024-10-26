# Introduction to Matplotlib for Image Processing


import numpy as np
import matplotlib.pyplot as plt

# --- Section 1: Plotting Basic Graphs ---
print("\n--- Section 1: Plotting Basic Graphs ---")

# Simple line plot
x = np.linspace(0, 10, 100)  # An array of 100 points from 0 to 10
y = np.sin(x)  # Sine of each point
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

input("Press Enter to continue...")
# --- End of Section 1 ---



# --- Section 2: Displaying Images with Matplotlib ---
print("\n--- Section 2: Displaying Images with Matplotlib ---")

image_data = np.random.rand(10, 10)  # 10x10 random image
plt.imshow(image_data, cmap='gray')  # Display in grayscale
plt.title("Random Grayscale Image")
plt.colorbar()  # Add a colorbar
plt.show()

input("Press Enter to continue...")
# --- End of Section 2 ---



# --- Section 3: Plotting Multiple Subplots ---
print("\n--- Section 3: Plotting Multiple Subplots ---")

x = np.linspace(0, 2 * np.pi, 400)
y1 = np.sin(x)
y2 = np.cos(x)
fig, (ax1, ax2) = plt.subplots(1, 2)  # 1 row, 2 columns of subplots
ax1.plot(x, y1)
ax1.set_title('Sine Wave')
ax2.plot(x, y2)
ax2.set_title('Cosine Wave')
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()

input("Press Enter to continue...")
# --- End of Section 3 ---



# --- Section 4: Customizing Plots ---
print("\n--- Section 4: Customizing Plots ---")

# Help:
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html

x = np.linspace(0, 10, 100)
y = np.exp(-x / 3) * np.sin(2 * np.pi * x)
plt.plot(x, y, color='red', linestyle='--', marker='o', label='Damped Sine') 
plt.title("Damped Sine Wave")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()  # Show the legend
plt.grid(True)  # Add a grid to the plot
plt.show()

input("Press Enter to continue...")
# --- End of Section 4 ---


print("\n--- End of Matplotlib Introduction ---")
