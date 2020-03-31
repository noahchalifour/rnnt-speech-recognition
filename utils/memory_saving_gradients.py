# The code for this came from 
# https://github.com/davisyoshida/tf2-gradient-checkpointing/blob/master/checkpointing.py

from functools import wraps
import random
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.eager import tape

def checkpointable(f):
    @wraps(f)
    def inner(*args, _checkpoint=False, _watch_vars=None, _force_seed=False, **kwargs):
        if _checkpoint:
            if _watch_vars is None:
                _watch_vars = []

            if _force_seed:
                seed = random.randint(1, 1<<31)

            watch_args = []

            flat_inputs = nest.flatten(args) + nest.flatten(list(kwargs.values()))
            flat_inputs = [x for x in flat_inputs if tf.is_tensor(x)]
            flat_inputs = [x for x in flat_inputs if x.dtype == tf.float32]
            unique_inputs = [x.deref() for x in set(x.experimental_ref() for x in flat_inputs)]

            unique_vars = [
                v.deref() for v in set(v.experimental_ref() for v in _watch_vars)
                if not any(v is inp for inp in flat_inputs)
            ]

            watches = unique_inputs + unique_vars
            tensor_watches = [tf.convert_to_tensor(x) for x in watches]

            with tape.stop_recording():
                if _force_seed:
                    tf.random.set_seed(seed)

                result = f(*args, **kwargs)

                flat_result = nest.flatten(result)
                # No idea what the point of this is but they do it in tf.custom_gradient so I'm doing it too
                flat_result = [tf.identity(x) for x in flat_result]
                output = nest.pack_sequence_as(result, flat_result)

            def grad(*output_grads):
                with tf.GradientTape() as g:
                    g.watch(watches)
                    if _force_seed:
                        tf.random.set_seed(seed)
                    recomputed_output = f(*args, **kwargs)
                    recomputed_output = [tf.identity(x) for x in nest.flatten(recomputed_output)]

                grads = g.gradient(recomputed_output, watches, output_gradients=output_grads)
                del g
                return grads

            tape.record_operation(str(f), flat_result, tensor_watches, grad)

            return output
        else:
            return f(*args, **kwargs)
    return inner