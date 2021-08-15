import numpy as np
import matplotlib.pyplot as plt
import time


def myConv(f, w):
    conv = np.zeros(len(f) + len(w) - 1)

    f = f.copy()
    w = w.copy()
    for i in range(len(f)):
        f.insert(0, 0)
        if i != 0:
            w.insert(0, 0)
    f.reverse()

    for n in range(len(conv)):
        dot_product = 0
        for x in range(n, len(f) + n):
            if x < len(w):
                dot_product += w[x] * f[x - n]
        conv[n] = dot_product

    return conv


Nfs = []
dtnps = []
dtmyconvs = []
dts = []

for N_f in range(1, 100):
    N_w = 100

    f = []
    w = []

    for nf in range(N_f):
        f.append(np.random.random())
    for nw in range(N_w):
        w.append(np.random.random())

    t1 = time.time()
    g_myConv = myConv(f, w)
    t2 = time.time()
    dt_myConv = t2 - t1
    print(f"myConv runtime: {dt_myConv}")

    t1 = time.time()
    g_numpy = np.convolve(f, w)
    t2 = time.time()
    dt_np = t2 - t1
    print(f"np.convolve runtime: {dt_np}")

    print(f"Difference in runtime between myConv and np.convolve: {dt_myConv - dt_np}")

    Nfs.append(N_f)
    dtnps.append(dt_np)
    dtmyconvs.append(dt_myConv)
    dts.append(dt_myConv - dt_np)

plt.figure()
plt.plot(Nfs, dtmyconvs, label='myConv')
plt.plot(Nfs, dtnps, label='np.convolve')
plt.legend()
plt.xlabel("$N_f$")
plt.ylabel("Runtime (s)")
plt.title("Runtime of myConv and np.convolve")
plt.savefig("1-3")

plt.show()
