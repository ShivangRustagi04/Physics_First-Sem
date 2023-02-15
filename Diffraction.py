import numpy as np
import matplotlib.pyplot as plt

def diffraction_pattern(wavelength, aperture_size, screen_distance, x_points):
    k = 2 * np.pi / wavelength
    x = np.linspace(-aperture_size / 2, aperture_size / 2, x_points)
    y = np.zeros_like(x)

    for i, x_i in enumerate(x):
        if abs(x_i) <= aperture_size / 2:
            y[i] = np.sin(k * x_i) ** 2

    return x, y * (2 / (wavelength * screen_distance))

wavelength = 500e-9  # 500 nm
aperture_size = 1e-3  # 1 mm
screen_distance = 1  # 1 meter
x_points = 1000

x, y = diffraction_pattern(wavelength, aperture_size, screen_distance, x_points)

plt.plot(x, y)
plt.xlabel("Position on screen (m)")
plt.ylabel("Intensity (a.u.)")
plt.title("Diffraction pattern")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def wave(x, t, k, L):
    return np.sin(2 * np.pi * k * (x - t))

def interference(x, t, L):
    k1 = 2.0 / L
    k2 = 4.0 / L
    return wave(x, t, k1, L) + wave(x, t, k2, L)

L = 1.0
x = np.linspace(-L, L, 1000)
t = 0
y = interference(x, t, L)

plt.plot(x, y)
plt.xlabel("Position (m)")
plt.ylabel("Amplitude (m)")
plt.title("Interference Pattern")
plt.show()
import numpy as np
import matplotlib.pyplot as plt

# Define the direction of the incident light
theta = np.pi/4

# Define the electric field vectors of the incident light
Ex_inc = np.cos(theta)
Ey_inc = np.sin(theta)

# Define the direction of the reflected light
theta_r = np.pi - theta

# Define the electric field vectors of the reflected light
Ex_ref = np.cos(theta_r)
Ey_ref = np.sin(theta_r)

# Plot the incident and reflected light
plt.quiver(0, 0, Ex_inc, Ey_inc, color='r', angles='xy', scale_units='xy', scale=1, label='Incident light')
plt.quiver(0, 0, Ex_ref, Ey_ref, color='b', angles='xy', scale_units='xy', scale=1, label='Reflected light')

# Add labels and legend
plt.xlabel('Ex')
plt.ylabel('Ey')
plt.legend()

# Show the plot
plt.show()
