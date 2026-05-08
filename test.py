import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Load an example .npy file
    # Ensure this path points to a valid .npy file in your processed_data folder
    npy_path = Path("processed_data/esteban_madera/thermal_npy/est_mad_test1_snap0001_0001s_thermal.npy")
    
    if not npy_path.exists():
        print(f"Error: Could not find the file {npy_path}")
        print("Please update the path to a valid .npy file.")
        return

    # Load the Numpy array
    thermal_array = np.load(npy_path)

    # Display the array as a grayscale image
    plt.imshow(thermal_array, cmap='gray')
    plt.colorbar(label='Temperature (°C)') # Add a colorbar to see the temperature scale
    plt.title(f"Thermal Image\n{npy_path.name}")
    plt.show()

if __name__ == "__main__":
    main()
