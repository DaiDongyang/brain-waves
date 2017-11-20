import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

a = sio.loadmat('./mats/sleep_data_row3_1.mat')
data_raw = a['data']
data = data_raw.reshape((-1, 1000))
fs = 200
N = 1000
f = 200 * np.arange(1, N+1)/N
S = np.fft.fft(data, n=1000, axis=1)
plt.plot(f, np.abs(S[0, :]))
plt.show()



# import scipy.io as sio
# import numpy.fft as fft
# import numpy as np
# import matplotlib.pyplot as plt
#
# # a = sio.loadmat('./mats/sleep_data_row3_1.mat')
# # b = a['data'].reshape((-1, 1000))
# # b = b[1:3, :]
# # # print(b)
# # # print(type(a))
# # c = fft.fft2(b)
# # N = 1000
# # fs = 200
# # n = np.arange(1, N+1)
# # f = fs * n / N
# # plt.plot(f, np.abs(c[0, :]))
# # plt.show()


# import matplotlib.pyplot as plt
# import plotly.plotly as py
# import numpy as np
#
# Fs = 150.0
# Ts = 1.0/Fs
# t = np.arange(0, 2, Ts)     # Try t = np.arange(0, 2, Fs)
#
# ff = 2
# y = np.sin(2*np.pi*ff*t)
# y2 = np.sin(2*np.pi*2*ff*t)
# ys = list()
# ys.append(y)
# ys.append(y2)
# ys = np.array(ys)
#
# n = len(y)
# k = np.arange(n)
# T = n/Fs
# frq = k/T
# frq = frq[0:int(n/2)]
#
# Ys = np.fft.fft(ys, axis=1)/n
# # Ys = Ys.T
#
#
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(t, ys[0, :])
# ax[1, 0].plot(frq, np.abs(Ys[0, 0:int(n/2)]))
# ax[0, 1].plot(t, ys[1, :])
# ax[1, 1].plot(frq, np.abs(Ys[1, 0:int(n/2)]))
# plt.show()





# print(ys.shape)
# print(Ys.shape)

# plt.plot(frq, np.abs(Y))
# Y = Y[int(range(n/2))]
# fig, ax = plt.subplots(2, 1)
# ax[0].plot(t, y)
# ax[0].set_xlabel('Time')
# ax[0].set_ylabel('Amplitude')
# ax[1].plot(frq, np.abs(Y), 'r')
# ax[1].set_xlabel('Freq(Hz)')
# ax[1].set_ylabel('|Y(freq)|')
# plot_url = py.plot_mpl(fig, filename='mpl-basic-fft')

# plt.show()
