
import sys
import platform
import tensorflow as tf

print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("Operating system:", platform.platform())
print("TensorFlow version:", tf.__version__)

print("\nAvailable GPUs:")
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected by TensorFlow.")

print("\nTensorFlow build information:")
print(tf.sysconfig.get_build_info())
