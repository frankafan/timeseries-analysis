import numpy as np
import matplotlib.pyplot as plt

R = 975
Nt = 100
dt = 0.00020
t = np.linspace(0, (dt * Nt), Nt)

H = np.ones(Nt)
H[0] = 0.5

D = np.zeros(Nt)
D[0] = 1 / dt

L = 5 * H


def St(R, L):
    return np.e ** (-R * t / L) * H


def Rt(R, L):
    return D - R / L * np.e ** (-R * t / L) * H


def RLresponse(R, L, V_in, dt):
    w = Rt(R, L)
    return np.convolve(w, V_in) * dt


plt.figure()
plt.plot(t, St(R, L), label='Theoretical S(t)')
plt.plot(t, RLresponse(R, L, H, dt)[slice(len(t))], '.', label='RLresponse')
plt.legend()
plt.xlim([0, 0.020])
plt.xlabel("Time (s)")
plt.ylabel("$V_{out}$")
plt.title("Theoretical vs. RLresponse $V_{out}$ with discretized $H(t)$ as input")
plt.savefig("2-1")

plt.figure()
plt.plot(t, Rt(R, L), label='Theoretical R(t)')
plt.plot(t, RLresponse(R, L, D, dt)[slice(len(t))], '.', label='RLresponse')
plt.legend()
plt.xlim([0, 0.020])
plt.xlabel("Time (s)")
plt.ylabel("$V_{out}$")
plt.title("Theoretical vs. Convolution $V_{out}$ with discretized $\delta(t)$ as input")
plt.savefig("2-2")

plt.show()
