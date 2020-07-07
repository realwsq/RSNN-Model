import matplotlib.pyplot as plt
import numpy as np

def raster_plot(ax,spikes,linewidth=0.8,max_spike = 10000, title="", **kwargs):

    n_t,n_n = spikes.shape
    event_times,event_ids = np.where(spikes)
    event_times = event_times[:max_spike]
    event_ids = event_ids[:max_spike]

    for n,t in zip(event_ids,event_times):
        ax.vlines(t, n - 0.5, n + 0.5, linewidth=linewidth, **kwargs)

    ax.set_xticks([0, n_t])
    ax.set_xlim([0, n_t])
    ax.set_xticklabels([0, n_t])
    ax.set_xlabel("time in ms")

    ax.set_ylim([-0.5, n_n-0.5])
    ax.set_yticks([0, n_n-1])
    ax.set_yticklabels([1, n_n])
    ax.set_ylabel("neuron number")
    
    if(title): ax.set_title(title)