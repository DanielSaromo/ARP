# keras_test.py - Auto-Rotating Perceptrons
# Python Example Code.
###############################################################################################################
# Author: Daniel Saromo.
###############################################################################################################
# Description:
#
# This code tests if the library was imported correctly.
#
###############################################################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from ARPkeras import AutoRotDense

model = Sequential()
model.add(Dense(20, input_shape=(123,)))
model.add(AutoRotDense(10, 0, 1, 4, activation='sigmoid'))

model.summary()

#import ARPkeras
#print(ARPkeras.__version__)
