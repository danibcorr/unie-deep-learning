# 3pps
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class RieszLayer(Layer):
    def __init__(self, output_channels, **kwargs):
        super().__init__(**kwargs)
        self.output_channels = output_channels

    def build(self, input_shape):
        _, _, _, input_channels = input_shape
        self.riesz_weights = self.add_weight(
            shape=(5 * input_channels, self.output_channels),
            initializer="glorot_uniform",
            trainable=True,
            name="riesz_weights",
        )
        super().build(input_shape)

    @tf.function
    def riesz_transform(self, input_image, H, W):
        # Recompute frequency grids based on current input dimensions
        n1 = tf.cast(
            tf.signal.fftshift(tf.linspace(-H // 2, H // 2 - 1, H)), tf.float32
        )
        n2 = tf.cast(
            tf.signal.fftshift(tf.linspace(-W // 2, W // 2 - 1, W)), tf.float32
        )

        n1 = tf.reshape(n1, (-1, 1))  # Column vector
        n2 = tf.reshape(n2, (1, -1))  # Row vector
        norm = tf.sqrt(n1**2 + n2**2 + 1e-8)

        real_part_R1 = n1 / norm
        imag_part_R1 = -tf.sqrt(1 - real_part_R1**2)
        real_part_R2 = n2 / norm
        imag_part_R2 = -tf.sqrt(1 - real_part_R2**2)

        # Fourier transform of the input
        I_hat = tf.signal.fft2d(tf.cast(input_image, tf.complex64))

        # First-order Riesz transforms
        I1 = tf.math.real(
            tf.signal.ifft2d(I_hat * tf.complex(real_part_R1, imag_part_R1))
        )
        I2 = tf.math.real(
            tf.signal.ifft2d(I_hat * tf.complex(real_part_R2, imag_part_R2))
        )

        # Second-order Riesz transforms
        I_20 = tf.math.real(
            tf.signal.ifft2d(I_hat * tf.complex(real_part_R1**2, imag_part_R1**2))
        )
        I_02 = tf.math.real(
            tf.signal.ifft2d(I_hat * tf.complex(real_part_R2**2, imag_part_R2**2))
        )
        I_11 = tf.math.real(
            tf.signal.ifft2d(
                I_hat
                * tf.complex(real_part_R1 * real_part_R2, imag_part_R1 * imag_part_R2)
            )
        )

        return tf.stack([I1, I2, I_20, I_11, I_02], axis=-1)

    def call(self, inputs):
        batch_size, H, W, input_channels = (
            tf.shape(inputs)[0],
            tf.shape(inputs)[1],
            tf.shape(inputs)[2],
            tf.shape(inputs)[3],
        )

        # Reshape for channel-wise processing
        inputs_reshaped = tf.reshape(inputs, (-1, H, W))  # Combine batch and channels

        # Apply Riesz transform to each input slice using vectorization
        riesz_transformed = tf.map_fn(
            lambda x: self.riesz_transform(x, H, W),
            inputs_reshaped,
            fn_output_signature=tf.float32,
        )

        # Reshape back to original batch format with Riesz feature dimension
        riesz_features = tf.reshape(
            riesz_transformed, (batch_size, H, W, 5 * input_channels)
        )

        # Linear combination using weights
        riesz_features_flat = tf.reshape(
            riesz_features, (batch_size * H * W, 5 * input_channels)
        )
        combined_features_flat = tf.matmul(riesz_features_flat, self.riesz_weights)
        combined_features = tf.reshape(
            combined_features_flat, (batch_size, H, W, self.output_channels)
        )

        return combined_features


# Verify functionality on a sample input with variable sizes
sample_input_small = np.random.rand(1, 14, 14, 1).astype(np.float32)
sample_input_large = np.random.rand(1, 56, 56, 1).astype(np.float32)

print("Small input shape:", sample_input_small.shape)
transformed_output_small = RieszLayer(output_channels=5)(sample_input_small)
print("Transformed output shape (small input):", transformed_output_small.shape)

print("Large input shape:", sample_input_large.shape)
transformed_output_large = RieszLayer(output_channels=5)(sample_input_large)
print("Transformed output shape (large input):", transformed_output_large.shape)


# Helper function to resize the dataset
def resize_dataset(images, target_size):
    resized_images = np.array(
        [cv2.resize(img, target_size, interpolation=cv2.INTER_AREA) for img in images]
    )
    return np.expand_dims(resized_images, axis=-1)


# Load and preprocess MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]

# Add channel dimension (grayscale)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Resize images for evaluation at different scales
x_test_small = resize_dataset(x_test[..., 0], (14, 14))
x_test_large = resize_dataset(x_test[..., 0], (56, 56))


# Define the neural network model using the functional API
def create_riesz_cnn(input_shape=(None, None, 1)):
    input_layer = tf.keras.Input(shape=input_shape)

    x = RieszLayer(16, name="riesz_1")(input_layer)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = RieszLayer(32, name="riesz_2")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = RieszLayer(40, name="riesz_3")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = RieszLayer(48, name="riesz_4")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    output = tf.keras.layers.Dense(10, activation="softmax")(x)

    return tf.keras.Model(inputs=input_layer, outputs=output)


# Create the Riesz-based CNN model
model = create_riesz_cnn(input_shape=(None, None, 1))

# Compile the model with AdamW optimizer
model.compile(
    optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()  # Corrected typo from 'suummary' to 'summary'

# Train the model with the learning rate scheduler
history = model.fit(
    x_train,
    y_train,
    shuffle=True,
    validation_split=0.2,
    batch_size=64,
    epochs=10,
)

# Evaluate the model on different scales
original_acc = model.evaluate(x_test, y_test, verbose=0)
small_acc = model.evaluate(x_test_small, y_test, verbose=0)
large_acc = model.evaluate(x_test_large, y_test, verbose=0)

# Print results
print(f"Accuracy on original scale (28x28): {original_acc[1]:.4f}")
print(f"Accuracy on smaller scale (14x14): {small_acc[1]:.4f}")
print(f"Accuracy on larger scale (56x56): {large_acc[1]:.4f}")
