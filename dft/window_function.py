import numpy as np
import matplotlib.pyplot as plt

# Define constants
T = 10
dt = 0.01


def boxcar(t, T):
    """Return the boxcar function, given t and T"""
    if 0 <= t <= T:
        return 1
    else:
        return 0


def hann(t, T):
    """Return the Hann function, given t and T"""
    if 0 <= t <= T:
        return 0.5 * (1 - np.cos(2 * np.pi * t / T))
    else:
        return 0


# Initialize lists for plotting
t_axis = []
boxcar_window = []
hann_window = []

t = 0
while t < T:  # Loop through window function lengths
    t_axis.append(t)
    # Store window functions over time
    boxcar_window.append(boxcar(t, T))
    hann_window.append(hann(t, T))
    t += dt

# Fourier transform window functions
boxcar_fft = np.fft.fftshift(np.fft.fft(boxcar_window) * dt)
hann_fft = np.fft.fftshift(np.fft.fft(hann_window) * dt)
# Create frequency domain
f_axis = np.fft.fftshift(np.fft.fftfreq(len(boxcar_window), dt))
# Create wavelength domain
w_axis = 2 * np.pi * f_axis

# Plot data
plt.figure()
plt.plot(t_axis, boxcar_window, label='Boxcar')
plt.plot(t_axis, hann_window, label='Hann')
plt.legend()
plt.xlabel('$t$')
plt.ylabel('$b(t)$')
plt.title("Window functions")
plt.savefig("Window functions")

plt.figure()
plt.plot(f_axis, np.real(boxcar_fft), label='Boxcar')
plt.plot(f_axis, np.real(hann_fft), label='Hann')
plt.legend()
plt.xlabel('$f$')
plt.ylabel('Real $B(f)$')
plt.title("Real part of Fourier transformed window functions")
plt.savefig("Real part of Fourier transformed window functions")

plt.figure()
plt.plot(f_axis, np.imag(boxcar_fft), label='Boxcar')
plt.plot(f_axis, np.imag(hann_fft), label='Hann')
plt.legend()
plt.xlabel('$f$')
plt.ylabel('Imaginary $B(f)$')
plt.title("Imaginary part of Fourier transformed window functions")
plt.savefig("Imaginary part of Fourier transformed window functions")

plt.figure()
plt.plot(f_axis, np.abs(boxcar_fft), label='Boxcar')
plt.plot(f_axis, np.abs(hann_fft), label='Hann')
plt.legend()
plt.xlabel('$f$')
plt.ylabel('$B(f)$ amplitude')
plt.title("Amplitude of Fourier transformed window functions")
plt.savefig("Amplitude of Fourier transformed window functions")

plt.show()
