import tensorflow as tf

# Create N-BEATSBlock custom layer
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self,
                input_size: int,
                theta_size: int,
                horizon: int,
                neurons: int,
                layers: int,
                **kwargs):
        super().__init__(**kwargs)
        
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.neurons = neurons
        self.layers = layers
    
        self.hidden = [tf.keras.layers.Dense(neurons, activation="relu") for _ in range(layers)]
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast


# Test our N-BEATS BLOCK class.
dummy_nbeats_block_layer = NBeatsBlock(
    input_size=7,
    theta_size=8,
    horizon=1,
    neurons=128,
    layers=4,
    name="???",
)

print(dummy_nbeats_block_layer(tf.expand_dims([1, 2, 3, 4, 5, 6, 7], axis=0)))
