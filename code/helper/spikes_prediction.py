## likelihood per spike
from scipy import signal
from scipy import misc
import numpy as np 

def sigmoid( signal ):
    # Prevent overflow.
    signal = np.clip( signal, -50, 50 )

    # Calculate activation signal
    signal = 1.0/( 1 + np.exp( -signal ))

    return signal

# one trial
def _calc_llh_of_spiking_time(voltages_scaled,sps_gt, n_rec, t_win, temperature_parameter):
    # temperature_parameter = 10
    spike_probability = sigmoid(voltages_scaled*temperature_parameter)
    spike_probability = signal.convolve2d(spike_probability,np.ones((t_win*2+1,1))/(t_win*2+1),mode='same')
    likelihood_per_neuron = np.zeros(n_rec)
    likelihoods = []
    for ni in range(n_rec):
        likelihood = spike_probability[:,ni][sps_gt[:,ni]==1]
        if len(likelihood):
          likelihood_per_neuron[ni] = np.mean(likelihood)
        likelihoods += [likelihood]
    return likelihood_per_neuron.mean(), likelihoods

def calc_likelihood_of_spiking_time_batch(voltages_scaled,sps_gt, n_rec, t_win=5, temperature_parameter=10):
    assert len(voltages_scaled) == len(sps_gt)
    llhs = [_calc_llh_of_spiking_time(v, sps, n_rec, t_win, temperature_parameter)[0] for (v, sps) in zip(voltages_scaled, sps_gt)]
    return np.mean(llhs)


