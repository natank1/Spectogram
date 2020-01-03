from keras.layers import Input,Reshape,Conv2D,add,multiply,Permute,Lambda,Dense
import keras.backend as K
from keras.models import Model
import numpy as np
def build_model( input_shape ,n_filter,n_hop):
    input_layer = Input(shape=(input_shape,))
    vec_dim = input_shape
    s_input = Reshape((vec_dim , 1, 1))(input_layer)

    sub_sample =(n_hop,1)
    output_real = Conv2D(n_filter, kernel_size=(3, 3), strides=sub_sample,
                         padding='same', trainable=False)(s_input)
    output_imag = Conv2D(n_filter, kernel_size=(3, 3), strides=sub_sample,
                         padding='same', trainable=False)(s_input)
    outpreal2 = multiply([output_real, output_real])
    outpimag2 =multiply([output_imag, output_imag])
    output = add([outpreal2, outpimag2])

    output = Permute((3, 1, 2))(output)

    log_spec = Lambda(lambda x: 10 * K.log(K.maximum(x, 1e-10)) / np.log(10.).astype(K.floatx()))(output)
    axis = tuple(range(K.ndim(output))[1:])
    # #
    # # # log_spec = log_spec - K.max(log_spec, axis=axis, keepdims=True)  # [-?, 0]
    # # # log_spec = K.maximum(log_spec, -1 * dynamic_range)  # [-80, 0]
    log_spec_0 = Lambda(lambda x: x - K.max(x, axis=axis, keepdims=True))(log_spec)
    log_spe_1 = Lambda(lambda x: K.maximum(x, -1 * 80.0))(log_spec_0)  # [-80, 0]
    bb = Permute((2, 1, 3))(log_spe_1)
    mm= Model(input_layer,bb)
    return mm

if __name__ =="__main__":
    mm =build_model(1000,1025,100)
    print (mm.summary())

