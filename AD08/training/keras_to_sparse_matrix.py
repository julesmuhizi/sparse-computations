import scipy.sparse as sparse
import numpy
import scipy


from keras_model import load_model
model = load_model('./model/ad08_0.9finSpar/model_ToyCar.h5')
keras_keys = ['q_dense_batchnorm',
 'q_dense_batchnorm_1',
 'q_dense_batchnorm_2',
 'q_dense_batchnorm_3',
 'q_dense_batchnorm_4',
 'q_dense']

for layer in keras_keys:
    if 'batchnorm' in layer:
        weights, biases = model.get_layer(layer).get_folded_weights()
    else:
        weights, biases = model.get_layer(layer).get_weights()
    quantizer = model.get_layer(layer).quantizers
    weights = sparse.csr_matrix(quantizer[0](weights))
    biases = quantizer[1](biases)
    scipy.io.mmwrite('./model/ad08_0.9finSpar/{}_weights'.format(layer), weights)
    scipy.io.mmwrite('./model/ad08_0.9finSpar/{}_bias'.format(layer), [biases])