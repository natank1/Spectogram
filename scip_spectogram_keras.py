import numpy as np
import tensorflow as tf
import keras
from keras.layers import Lambda,Input,Dense,Permute,RepeatVector,Subtract,Reshape
from keras.models import Model,load_model,Sequential
import keras.backend as K
import math
hamming_const0= 0.0008878628054811652

def bring_mat (feat_dim):
    a = np.zeros((feat_dim, feat_dim), float)
    np.fill_diagonal(a, hamming_const0)
    return a
def create_model(nb_times,feat_dim, model_ext):
    mat_scale =bring_mat(feat_dim)
    z00= Input (shape=(nb_times,feat_dim))
    zz = Dense(feat_dim, use_bias=False, trainable=False, weights=[mat_scale], activation="linear")(z00)
    zz = Reshape((nb_times,feat_dim, 1))(zz)
    model_ext.layers.pop(0)
    modela = model_ext(zz)
    newModel = Model(z00, modela)
    newModel.compile(loss='binary_crossentropy', optimizer="RMSprop", metrics=['accuracy'])
    return newModel



def hamming_window (frame_len,sample_rate):
    denom=frame_len-1
    hamming = np.asarray([0.54 - 0.46 * np.cos(2*np.pi * jj / denom) for jj in range(frame_len)])

    scal = 1 / math.sqrt(sum([i * i for i in hamming]) * 8000)
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

def prepare_spectogram_model (signal_len,frame_length,frame_hop,derive_math, hamm_scale, ham_diag,nfft, use_derive=True):
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

        x5= Lambda(lambda x: tf.signal.rfft(x, fft_length=[nfft]))(x4)
        x6 =Lambda(lambda  x: hamm_scale*K.abs(x))(x5)
        model_00= Model(inputs=x0, outputs=x6)

        return model_00


def prepare_spectogram_model_with_constant (signal_len, frame_length, frame_hop, derive_math,  ham_diag,nfft, use_derive=True):
    x0 = Input(shape=(signal_len,))

    if use_derive:
        x00 = Dense(signal_len - 1, use_bias=False, trainable=False, weights=[derive_math], activation="linear")(x0)
    else:
        x00 = x0

    x1 = Lambda(lambda x: tf.signal.frame(x, frame_length, frame_hop))(x00)

    x2 = Lambda(lambda x: K.mean(x, axis=-1))(x1)
    x2 = RepeatVector(frame_length)(x2)
    x2 = Permute((2, 1))(x2)
    x3 = Subtract()([x1, x2])

    x4 = Dense(frame_length, use_bias=False, trainable=False, weights=[ham_diag], activation="linear")(x3)

    x5 = Lambda(lambda x: tf.signal.rfft(x, fft_length=[nfft]))(x4)
    xconst = Input(shape=(1, 1))
    x6 = Lambda(lambda x: xconst * K.abs(x))(x5)

    model_00 = Model(inputs=[x0,xconst], outputs=x6)

    return model_00
if __name__ == '__main__':


    signal_len = 25000
    frame_len = 400
    sample_rate = 8000
    frame_hop = 160
    nfft=2048
    hamming_scale, hamming_math = hamming_window(frame_len, sample_rate)


    my_path = "C:\\myfile25000.npy"
    modelb = prepare_spectogram_model_with_constant(24999, frame_len, frame_hop, [],  hamming_math,nfft,    use_derive=False)

    # Prepare Input
    z0 = np.load(my_path)
    z0 = z0[0, 1:] - 0.97 * z0[0, :-1]
    z0 = np.expand_dims(z0, axis=0)

    # prepare constant
    tt = np.array([4.0])
    tt = np.expand_dims(tt, axis=1)
    tt = np.expand_dims(tt, axis=2)

    #Actual prediction
    y1= modelb.predict([z0,tt])
    print (y1[0,0,:5])

    #Same mode with no constant
    modela = prepare_spectogram_model(24999, frame_len, frame_hop, [], 4., hamming_math,nfft,    use_derive=False)
    y1 = modela.predict([z0])
    print (y1[0,0,:5])
    exit(33)

