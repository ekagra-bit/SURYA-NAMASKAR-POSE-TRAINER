import sys
import types

# 👉 TF-DF ko fake module se override karo (import fail na ho)
sys.modules['tensorflow_decision_forests'] = types.ModuleType('tensorflow_decision_forests')

import tensorflow as tf
import tensorflowjs as tfjs

# SavedModel load karo (tumne abhi train karke banaya hai)
model = tf.keras.models.load_model("saved_model", compile=False)

# TFJS me convert
tfjs.converters.save_keras_model(model, "web_model_clean")

print("DONE")