import numpy as np
from keras.layers import Lambda,Input,Subtract,Dense
from keras.models import Model
import keras.backend as K

def keras_pre_emp(sig_len,pre_emp_c =0.97):

    in0 = Input(shape=(sig_len,))
    x1= Lambda (lambda x: x[:,1:])(in0)
    x2= Lambda (lambda x: x[:,:-1])(in0)
    x3=Lambda(lambda  x: 0.97*x)(x2)
    x4= Subtract()([x1,x3])

    model =Model(in0,x4)
    return model


" The following functions come together"
def create_pre_emp_mat(signal_len, pre_emp_c=0.97):
    vector_dim =signal_len-1
    a = np.zeros((vector_dim, vector_dim), float)
    np.fill_diagonal(a, pre_emp_c)

    b = np.identity(vector_dim)
    b = np.append(np.zeros((vector_dim, 1), float), b, axis=1)
    b1 = np.append(a, np.zeros((vector_dim, 1), float), axis=1)
    c = b - b1
    weight_mat = np.transpose(c)
    return weight_mat



def create_pre_emp(signal_len, pre_emp_c=0.97):
    in0 = Input(shape=(signal_len,))
    w_mat=create_pre_emp_mat(signal_len, deriv_coeff=0.97)
    x00 = Dense(signal_len - 1, use_bias=False, trainable=False, weights=[w_mat], activation="linear")(in0)
    model= Model(in0,x00)
    return model

if __name__=="__main__":
    xx = np.random.rand(1000)
    print ("linear process no keras")
    zz1=xx[1:]-0.97*xx[:-1]

    "keras with  dense layer"
    model =create_pre_emp(1000)
    zz= model.predict(np.expand_dims(xx,axis=0))

    "keras_no dense layer"
    model =keras_pre_emp(1000)
    zz2 = model.predict(np.expand_dims(xx, axis=0))
