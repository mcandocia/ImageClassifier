import full_neural_network as fnn
import math
import prepare_image_data
import numpy as np
#200 is fine

#long network


#actual parameters
batch_size_c = 40
kernel_sizes_c = [3,32,40,45,50,55,10]
input_dimensions_c = [200,264]
convolution_dimensions_c = [(13,13),(9,9),(7,7),(5,5)]
pool_sizes_c = [(4,4),(2,2),(2,2),(2,2)]
stride_sizes_c = [(2,2),(2,2),(2,2),(2,2)]
layer_pattern_c = ['I','C','C','C','C','F','F','O']
relu_pattern_c = [False,True,True,True,False,False,False]
dropout_rate_c = [0.5,0.5,0.5,0.5,0.5,0.3,0.3,0.25]
rng_seed_c = 447
base_learning_rate_c = 0.116
momentum_c = 0.9
learning_decay_per_epoch_c = 0.92
name_c = 'test_model'
param_index_c = 12#continuing 10 with half L2 norm
address_c = '/home/max/workspace/StateFarmDistract/'
l2_norm_c = 0.000015

#NOTE: divided L2 norm by 3 to allow better generalization
def main():

    '''network = fnn.neural_network(batch_size = batch_size_c,
kernel_sizes = kernel_sizes_c, input_dimensions = input_dimensions_c,
convolution_dimensions = convolution_dimensions_c,pool_sizes = pool_sizes_c,
stride_sizes = stride_sizes_c, layer_pattern = layer_pattern_c,
relu_pattern = relu_pattern_c,dropout_rate = dropout_rate_c,
rng_seed = rng_seed_c, base_learning_rate = base_learning_rate_c,
momentum = momentum_c, learning_decay_per_epoch = learning_decay_per_epoch_c,
name=name_c,param_index = param_index_c,address = address_c,l2_norm = l2_norm_c)
'''
    print 'created network'
    network_filename = 'test_model_inetwork_12.pickle'
    network = fnn.load_network_isolate(network_filename)
    network.l2_norm = l2_norm_c
    #network.learning_rate_raw = 0.001
    network.L2_penalty.set_value(np.float32(network.l2_norm))
    print 'loaded network'
    #network.name = 'test_model_reloaded'
    #network.learning_rate_raw = 0.006
    #network.param_index = 9
    network.run_iterations()
    #return 0
    #fetcher = prepare_image_data.fetcher(10)
    #image_array = fetcher.fetch_validation()[0]
    #print type(image_array)
    #results = network.predict(image_array)
    #return results
    #print 'somehow managed to survive the iterations...'
    network.save_network(mode='hallelujah')
    return 0

if __name__=='__main__':
    main()
