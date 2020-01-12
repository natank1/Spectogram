# Spectogram 

This project contains some solutions for programming specotgram functions using keras layers. The main purpose was to overcome early feature 
extraction in speech and speech spoofing assignments. We mimic scipy.signal using hamming and some other implemtations
In more details:

pre_emp.py: 
It suggests two methdologies for perfroming pre emp to mono signal :
  1 Using a regular dense layer (very "keras")
  2 By regular subtraction 
  
Kapre
We aim to mimic kapre package using keras generic layers https://github.com/keunwoochoi/kapre/. The motivation for this step is our need
 to perfrom time_frequency spectogram  in Keras.Net We thus needed to use generic layers (keras.net is not familiar with Kapre)
 The output of the file is a spectogram that one can connect to a regular Conv2 Layer
 
