import numpy as np
import matplotlib.pyplot as plt

# Define constants
dt = 1e-3
t_H1 = 10
t_H2 = 30
domain = [-80, 80]


def g(t, t_H):
    """Return the Gaussian function, given t and t_H"""
    return (1 / (np.pi ** 0.5 * t_H)) * np.exp(-(t / t_H) ** 2)


def G(w, t_H):
    """Return the analytical Fourier transform of the Gaussian function, given w and t_H"""
    return np.exp(-w ** 2 * t_H ** 2 / 4)


# Initialize lists for plotting Gaussian functions
g1 = []
g2 = []
t = domain[0]
while t < domain[-1]:  # Loop through time domain
    # Store Gaussian functions over time
    g1.append(g(t, t_H1))
    g2.append(g(t, t_H2))
    t += dt

# Fourier transform Gaussian functions
G_fft1 = np.fft.fftshift(np.fft.fft(g1) * dt)
G_fft2 = np.fft.fftshift(np.fft.fft(g2) * dt)
# Create frequency domain
f_axis = np.fft.fftshift(np.fft.fftfreq(len(g1), dt))
# Create wavelength domain
w_axis = 2 * np.pi * f_axis

# Initialize lists for plotting analytical Fourier transforms of Gaussian functions
G_anal1 = []
G_anal2 = []
for w in w_axis:  # Loop through wavelength
    # Store Fourier transforms of Gaussian functions over wavelengths
    G_anal1.append(G(w, t_H1))
    G_anal2.append(G(w, t_H2))

# Plot data
plt.figure()
plt.plot(np.linspace(domain[0], domain[-1], len(g1)), g1, label='$t_H=10$')
plt.plot(np.linspace(domain[0], domain[-1], len(g2)), g2, label='$t_H=30$')
plt.legend()
plt.xlabel('t')
plt.ylabel('g(t)')
plt.title("g(t)")
plt.savefig("g(t)")

plt.figure()
plt.plot(w_axis, np.abs(G_fft1), label='$t_H=10$ Numpy')
plt.plot(w_axis, np.abs(G_anal1), label='$t_H=10$ Analytical')
plt.plot(w_axis, np.abs(G_fft2), label='$t_H=30$ Numpy')
plt.plot(w_axis, np.abs(G_anal2), label='$t_H=30$ Analytical')
plt.legend()
plt.xlabel('$\omega$')
plt.ylabel('$G(\omega$)')
plt.title("G(t) amplitude")
plt.savefig("G(t) amplitude")

plt.figure()
plt.plot(w_axis, np.real(G_fft1), label='$t_H=10$ Numpy')
plt.plot(w_axis, np.real(G_anal1), label='$t_H=10$ Analytical')
plt.plot(w_axis, np.real(G_fft2), label='$t_H=30$ Numpy')
plt.plot(w_axis, np.real(G_anal2), label='$t_H=30$ Analytical')
plt.legend()
plt.xlabel('$\omega$')
plt.ylabel('$G(\omega$)')
plt.title("Real G(t)")
plt.savefig("Real G(t)")

plt.figure()
plt.plot(w_axis, np.imag(G_fft1), label='$t_H=10$ Numpy')
plt.plot(w_axis, np.imag(G_anal1), label='$t_H=10$ Analytical')
plt.plot(w_axis, np.imag(G_fft2), label='$t_H=30$ Numpy')
plt.plot(w_axis, np.imag(G_anal2), label='$t_H=30$ Analytical')
plt.legend()
plt.xlabel('$\omega$')
plt.ylabel('$G(\omega$)')
plt.title("Imaginary G(t)")
plt.savefig("Imaginary G(t)")

plt.show()
