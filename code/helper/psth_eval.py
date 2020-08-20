from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

PSTH_sigma = 10

def calc_psth_one_stim(spike_trains, filter=True, sigma=10):
  # spike_trains: [n_rep/n_neuron, T]
  averaged_ST = np.mean(spike_trains, 0)
  if filter:
    return gaussian_filter1d(averaged_ST, sigma)
  else:
    return averaged_ST

def calc_psth_all_stims(spike_trains, filter=True, sigma=10):
  # spike_trains: [n_stim, n_neuron, T]
  psth_all_stims = np.array([calc_psth_one_stim(ST, filter, sigma) for ST in spike_trains])
  return np.mean(psth_all_stims, 0)



def calc_r(psth1, psth2, delay=input_delay):
    return pearsonr(psth1[input_delay:], psth2[input_delay:])[0]

# def calc_explainable_variability(psth_gt):
#     # TODO

# def calc_explainable_variability_predicted(psth_predict, psth_gt):
#     explainable_variablity = calc_explainable_variability(psth_gt)
#     r = calc_r(psth_predict, psth_gt)
#     return r/explainable_variablity


def PSTH_viz(ax, psth_gt, psth_predict):
    # spikes: [T, n_neurons]

    n_t = range(sps.shape[1])
    ax.plot(n_t, psth_gt, 'r', label="gt")
    ax.plot(n_t, psth_predict, 'g', label='RSNN')
    # ax.title('population psth')
    ax.legend()
    ax.show()


if __name__ == "__main__":
	'''
	sps: (ground truth) [trial; T; neuron]
	spikes_np: (prediction) [trial; T; neuron]
	'''
    psth_gt = calc_psth_all_stims([ST.T for ST in sps], filter=True, sigma=psth_sigma)
	psth_predict = calc_psth_all_stims([ST.T for ST in spikes_np], filter=True, sigma=psth_sigma)
	print("r=%f"%(calc_r(psth_gt, psth_predict)))

	fig,ax = plt.subplots(1, figsize=(10,2))
	PSTH_viz(ax, psth_gt, psth_predict)


