import obspy
from obspy import UTCDateTime
import numpy as np

import os, sys
sys.path.insert(1, os.getcwd())
from internal.feature_extraction import features
from internal.visualization import plot
    
path = "./data/9F.NUPH..*.D.2021.143"
st = obspy.read(path)
st.trim(starttime = UTCDateTime("2021-05-23T12:00:00"), endtime = UTCDateTime("2021-05-23T16:00:00"))
# st.resample(.1)
st.detrend("demean")
# st.plot(block=True)
# quit()


n_channels = len(st)

SR = np.zeros(n_channels)
signals = []

for i, tr in enumerate(st):
    signals.append(tr.data)
    SR[i] = tr.stats.sampling_rate
x = None

# quit()


i = 1
print(st[i].stats.channel)
win_length = 3600 #s
win_samples = int(win_length * SR[i])

time, freq, Sxx = features.spectrogram(signals[i], SR[i], win_samples)
fig, ax, _ = plot.spectrogram(time, freq, Sxx, logscale=True)

# ax.set_ylim(0,.1)
fig.show()

fig, ax = plot.signal(np.arange(len(st[i])), st[i])

fig.show()
input()




