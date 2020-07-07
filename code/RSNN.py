%tensorflow_version 2.x
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import datetime
import io
import os



def pseudo_derivative(v_scaled, dampening_factor):
  return dampening_factor * tf.maximum(0.,1 - tf.abs(v_scaled))

@tf.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):

    # This means: z = 1 if v > thr else 0
    z_ = tf.greater(v_scaled, 0.)
    z_ = tf.cast(z_, dtype=tf.float32)

    def grad(dy):
        # This is where we overwrite the gradient
        # dy = dE/dz (total derivative) is the gradient back-propagated from the loss down to z(t)
        dE_dz = dy
        dz_dv_scaled = pseudo_derivative(v_scaled, dampening_factor)
        dE_dv_scaled = dE_dz * dz_dv_scaled

        return [dE_dv_scaled, tf.zeros_like(dampening_factor)]

    return tf.identity(z_, name="SpikeFunction"), grad


@tf.function
def L1_loss(weights):
    reg_loss = tf.reduce_sum(tf.abs(weights))
    # reg_loss = tf.reduce_sum((weights)**2)
    return reg_loss

@tf.function
def classification_loss(v_scaled, spike_labels, temperature_parameter):
    # temperature_parameter is either trained or tuned by hand
    spike_logits = temperature_parameter * v_scaled
    spike_probability = tf.sigmoid(spike_logits)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(spike_labels,spike_logits)
    loss_reduced = tf.reduce_mean(loss)

    return loss_reduced, spike_probability, loss

# NB: There could be other interesting loss functions
# Like convolve the label_spikes with a gaussian of variance 4 ms
# and derectly optimize || spikes - spike_labels_concolved ||^2


class RSNN(tf.keras.layers.Layer):
    def __init__(self, num_neurons, thr, spike_delay, input_delay, tau=20., dampening_factor=0.3, ground_truth=False):
        super().__init__()
        
        self.num_neurons = num_neurons
        # self.state_size = (num_neurons, num_neurons) # weight variable
        self.state_size = (num_neurons, num_neurons*spike_delay, num_neurons*input_delay) # weight variable
        self.decay = tf.exp(-1 / tau)

        self.dampening_factor = dampening_factor
        self.thr = thr

        self.input_weights = None
        self.recurrent_weights = None
        self.disconnect_mask = None

        self.post_spike_delay = spike_delay
        self.input_delay = input_delay

        self.ground_truth = ground_truth

    def build(self, input_shape):
        if self.ground_truth:
            n_in = input_shape[-1] - self.num_neurons # first n_in inputs: image; 
                      # last num_neurons inputs: spike history of last time step
        else:
            n_in = input_shape[-1]
            
        n = self.num_neurons
        self.num_inputs = n_in
 
        rand_init = tf.keras.initializers.RandomNormal
        const_init = tf.keras.initializers.Constant


        # define the input weight variable
        # self.input_weights_spatial = self.add_weight(
        #     shape=(n_in,n),
        #     trainable=True,
        #     initializer=rand_init(stddev=1. / np.sqrt(n_in)),
        #     name='input_weights_spatial')
        
        # self.input_weights_temporary = self.add_weight(
        #     shape=(self.input_delay,),
        #     trainable=True,
        #     initializer= const_init(1.  / np.sqrt(self.input_delay)),
        #     name='input_weights_temporary')
        self.input_weights = self.add_weight(
            shape=(n_in,n, self.input_delay),
            trainable=True,
            initializer=rand_init(stddev=1. / np.sqrt(n_in * self.input_delay)),
            name='input_weights_spatial')
        
        # define the recurrent weight variable
        self.disconnect_mask = tf.cast(np.repeat(np.diag(np.ones(n, dtype=np.bool))[:, :, np.newaxis],self.post_spike_delay,axis=2), tf.bool)
        self.recurrent_weights = self.add_weight(
            shape=(n, n, self.post_spike_delay),
            trainable=True,
            initializer=rand_init(stddev=1. / np.sqrt(n * self.post_spike_delay)),
            name='recurrent_weights')

        super().build(input_shape)

    def get_recurrent_weights(self): # get the coupled weights
        w_rec = tf.where(self.disconnect_mask, tf.zeros_like(self.recurrent_weights), self.recurrent_weights)
        return w_rec

    def get_input_weights(self):
        # w_in = tf.expand_dims(self.input_weights_spatial,2)*self.input_weights_temporary
        w_in = self.input_weights
        return w_in

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        initialized_state = (tf.zeros((batch_size,self.num_neurons)), 
                             tf.zeros((batch_size,self.num_neurons, self.post_spike_delay)), 
                             tf.zeros((batch_size,self.num_inputs, self.input_delay)))
        return initialized_state

    def call(self, inputs, states, constants=None):

        # spike_buffer is a tensor of shape: batch (b) x neuron (i) x delay (k)
        # weight has shape: pre-neuron (i) x post_neuron (j) x delay (k)
        #
        # we want to compute the sum:
        # i(b,j) = sum_i sum_k spike_buffer(b,i,k) W(i,j,k)


        # NB: this is great idea,
        # I think when some boolean "self.ground_truth" is True
        # One should replace states[1] (or which ever is the spike)
        # by the spikes entry in the inputs.
        # Otherwise do not use the ground truth spikes from the inputs
        # TODO: add buffer
        # if self.ground_truth:
            # old_z = inputs[:,self.num_inputs:]

        spike_buffer = states[1]
        w_rec = self.get_recurrent_weights()
        i_from_spike_buffer = tf.einsum("bik,ijk->bj",spike_buffer, w_rec)
      
        # the buffer update of the spike buffer should be like:
        # new_buffer = concat([buffer[1:],new_spike])


        # same for the inputs as:
        # input_buffer is a tensor of shape: batch (b) x n_in (i) x delay (k')
        # weight has shape: n_in (i) x neuron (j) x delay (k')
        input_buffer = states[2]
        video_inputs = inputs[:,:self.num_inputs]
        input_buffer = tf.concat([input_buffer[:,:,1:],tf.expand_dims(video_inputs,2)], axis=2)
        w_in = self.get_input_weights()
        i_from_in_buffer = tf.einsum("bik,ijk->bj", input_buffer, w_in)

        ### to do exponential filters:
        ### alpha_vector = [exp(-dt/tau_1), exp(-dt/ tau_2), exp(-dt/tau3)]
        ### new_exp_buffer = alpha_vector * exp_buffer  + (1 - alpha_vector) new_spike
        ### 
        ### per  connection we have exp_buffer_weights:
        ### if you have rec_exp_buffer_weights has shape:
        ### pre-neuron (i) x post_neuron (j) x delay (k')

        ### you can compute the same weight
        ### i_from_exp_buffer = tf.einsum("bik,ijk->bj",exp_buffer,rec_exp_buffer_weights)
      

        i_in = i_from_spike_buffer + i_from_in_buffer

        # update the voltage
        old_v = states[0]
        old_z = spike_buffer[:,:,-1]
        d = self.decay
        # NB: this implements the reset, after a spike, the voltage should decrease
        i_reset = - self.thr * old_z
        new_v = d * old_v + (1-d) * i_in + i_reset

        # NB: v_scaled is important and should be one of the output of the model
        new_v_scaled = (new_v - self.thr) / self.thr
        new_z = SpikeFunction(new_v_scaled, self.dampening_factor)
        new_spike_buffer = tf.concat([spike_buffer[:,:,1:],tf.expand_dims(new_z, 2)], axis=2)

        new_state = (new_v, new_spike_buffer, input_buffer)
        
        return (new_z, new_v, new_v_scaled), new_state

        
def build_model(n_in, n_rec, thr=0.03, spike_delay=100, input_delay=40):
    cell = RSNN(n_rec, thr, spike_delay, input_delay) 
    cell.build([None,None,n_in])
    rnn = tf.keras.layers.RNN(cell,return_sequences=True)

    return rnn, cell