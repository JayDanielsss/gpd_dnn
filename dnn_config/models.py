"""
Main DNN model for Re[H] and Im[H].

File initialized: 20260310
Last modified: 20260310
"""

import tensorflow as tf
from .losses import SimultaneousObservablesLoss
 
_INITIALIZER_MINIMUM_VALUE = -0.1
_INITIALIZER_MAXMIMUM_VALUE = 0.1

_NUMBER_NODES_HIDDEN_1 = 10
_NUMBER_NODES_HIDDEN_2 = 10
_NUMBER_NODES_HIDDEN_3 = 10
_NUMBER_NODES_HIDDEN_4 = 10

NUMBER_OF_REPLICAS = 100

def cff_h_model():
    initializer = tf.keras.initializers.RandomUniform(
        minval = _INITIALIZER_MINIMUM_VALUE, 
        maxval = _INITIALIZER_MAXMIMUM_VALUE,
        seed = None)
    
    all_model_inputs = tf.keras.Input(
        shape = (4,), # expect k, xB, t, Q^{2}
        name = "input_values")
     
    kinematic_inputs = tf.keras.layers.Lambda(
        lambda x: x[:, :3],
        name = 'input_kinematics')(all_model_inputs) # takes xB, t, Q^{2}

    # [NOTE]: ONLY kinematic inputs goes into the DNN!
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_1, kernel_initializer = initializer, activation = "relu")(kinematic_inputs)
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_2, kernel_initializer = initializer, activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_3, kernel_initializer = initializer, activation = "relu")(hidden)
    hidden = tf.keras.layers.Dense(_NUMBER_NODES_HIDDEN_4, kernel_initializer = initializer, activation = "relu")(hidden)

    cff_outputs = tf.keras.layers.Dense(
        2, # Re[H] and Im[H]
        activation = "linear", 
        name = "cff_h")(hidden)
    
    full_model_outputs = tf.keras.layers.Concatenate(
        name = "kinematics_and_cffs")([cff_outputs, all_model_inputs])
    
    model = tf.keras.Model(
        inputs = all_model_inputs,
        outputs = full_model_outputs)

    model.compile(
        optimizer = tf.keras.optimizers.Adam(), 
        loss = SimultaneousObservablesLoss(),
        metrics = ["mae", "mse"])
    return model