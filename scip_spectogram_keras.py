import numpy as np
import tensorflow as tf
import keras
from keras.layers import Lambda,Input,Dense,Permute,RepeatVector,Subtract
from keras.models import Model
import keras.backend as K
import math
def hamming_window (frame_len,sample_rate):
    denom=frame_len-1
    hamming = np.asarray([0.54 - 0.46 * np.cos(2*np.pi * jj / denom) for jj in range(frame_len)])
    ll = sum([i * i for i in hamming])
    print(1 / math.sqrt(ll * sample_rate))
    frame_length = 400
    scal = 1 / math.sqrt(sum([i * i for i in hamming]) * 8000)
    print(scal)
    print(hamming.shape, hamming)
    ham_diag = np.diag(hamming)
    return scal, ham_diag


def create_derivative_layer(signal_len, deriv_coeff=0.97):
    vector_dim =signal_len-1
    a = np.zeros((vector_dim, vector_dim), float)
    np.fill_diagonal(a, deriv_coeff)

    b = np.identity(vector_dim)
    b = np.append(np.zeros((vector_dim, 1), float), b, axis=1)
    b1 = np.append(a, np.zeros((vector_dim, 1), float), axis=1)
    c = b - b1
    weight_mat = np.transpose(c)
    return weight_mat

def prepare_spectogram_model (signal_len,frame_length,frame_hop,derive_math, hamm_scale, ham_diag, use_derive=True):
        x0= Input(shape=(signal_len,))
        if use_derive:
            x00 = Dense(signal_len-1,use_bias=False,trainable=False,weights=[derive_math],activation="linear")(x0)
        else:
            x00= x0
        x1= Lambda(lambda x: tf.signal.frame(x, frame_length, frame_hop))(x00)
        x2= Lambda(lambda x: K.mean(x,axis=-1))(x1)
        x2= RepeatVector(frame_length)(x2)
        x2 =Permute((2,1))(x2)
        x3 = Subtract()([x1,x2])

        x4= Dense (frame_length ,use_bias=False, trainable=False,weights=[ham_diag],activation="linear")(x3)
        x5= Lambda(lambda x: tf.signal.rfft(x, fft_length=[2048]))(x4)
        x6 =Lambda(lambda  x: hamm_scale*K.abs(x))(x5)
        model_00= Model(inputs=x0, outputs=x6)
        # print (m12.summary())
        return model_00

if __name__ == '__main__':
    signal_len = 12000
    frame_len=400
    sample_rate=8000
    frame_hop=240
    my_path ="my_path.npy"
    z0 = np.load(my_path)
    weight_derive_mat = create_derivative_layer(signal_len)
    hamming_scale, hamming_math=  hamming_window(frame_len, sample_rate)
    model = prepare_spectogram_model(signal_len, frame_len, frame_hop,     weight_derive_mat, hamming_scale, hamming_math,use_derive=True  )


    # model.save("trial.h5")

    # y1 =m12.predict(np.expand_dims(z1,axis=0))
    y1 = model.predict(z0)

    print("tttt ", y1[0, 0, 0:5])
    exit(33)

