import tensorflow as tf


def rnnt_loss(
    logits: tf.Tensor,
    labels: tf.Tensor,
    logit_length: tf.Tensor,
    label_length: tf.Tensor,
    blank_id: int = 0,
) -> tf.Tensor:
    """
    Computes the RNN Transducer loss from https://arxiv.org/pdf/1211.3711.

    Args:
        logits: A tensor of shape `(batch_size, max_input_length, max_label_length + 1, vocab_size)`.
                These are the raw outputs from the joint network.
        labels: A tensor of shape `(batch_size, max_label_length)` containing the integer labels.
        logit_length: A tensor of shape `(batch_size,)` containing the true lengths of the input sequences.
        label_length: A tensor of shape `(batch_size,)` containing the true lengths of the label sequences.
        blank_id: The integer identifier for the blank symbol. Defaults to 0.

    Returns:
        A tensor of shape `(batch_size,)` containing the loss value for each item in the batch.
    """
    # Get the dimensions of the input tensors
    batch_size = tf.shape(logits)[0]
    max_T = tf.shape(logits)[1]
    max_U = tf.shape(logits)[2] - 1

    # Convert logits to log-probabilities for numerical stability
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Create the DP table `alpha` initialized with a large negative number instead of -inf
    # to avoid numerical instability in gradient computation
    # Shape: (batch_size, max_input_length, max_label_length + 1)
    alpha = tf.ones([batch_size, max_T + 1, max_U + 1]) * -float("inf")

    # Initialize alpha(0, 0) = 0 for all items in the batch.
    # This represents the initial state with log-probability 1 (log(1)=0)
    batch_indices = tf.expand_dims(
        tf.cast(tf.range(batch_size), dtype=tf.int32), axis=1
    )
    zeros_indices = tf.zeros((batch_size, 2), dtype=tf.int32)
    indices = tf.concat([batch_indices, zeros_indices], axis=1)
    updates = tf.zeros([batch_size])
    alpha = tf.tensor_scatter_nd_update(alpha, indices, updates)

    # print(f"Alpha: {alpha.numpy()}")

    # Wrap the main loops in tf.function for performance
    @tf.function
    def compute_alpha():
        # Create a mutable TensorArray for our DP table
        alpha_TBU = tf.transpose(alpha, perm=[1, 0, 2])
        alpha_ta = tf.TensorArray(tf.float32, size=max_T + 1, clear_after_read=False)
        alpha_ta = alpha_ta.unstack(alpha_TBU)

        # Iterate over each time step `t`
        for t in tf.range(1, max_T + 1):
            # Get the alpha values from the previous time step
            prev_alpha_t = alpha_ta.read(t - 1)
            inner_ta = tf.TensorArray(
                tf.float32, size=max_U + 1, clear_after_read=False
            )

            # Iterate over each label step `u`
            for u in tf.range(max_U + 1):
                # Path 1: From the top (t-1, u), consuming an input and emitting a blank
                p_blank = log_probs[:, t - 1, u, blank_id]
                log_alpha_blank = prev_alpha_t[:, u] + p_blank

                if u == 0:
                    # If we are at the first label position, we can only come from the top
                    inner_ta = inner_ta.write(0, log_alpha_blank)
                else:
                    # Path 2: From the left (t, u-1), emitting a label
                    log_alpha_from_left_prev = inner_ta.read(u - 1)

                    # Gather the log-probabilities of the true labels
                    batch_indices = tf.range(batch_size, dtype=tf.int64)
                    label_indices = tf.cast(labels[:, u - 1], dtype=tf.int64)

                    # Create indices for gathering from log_probs
                    # Shape of indices: (batch_size, 4)
                    indices_to_gather = tf.stack(
                        [
                            batch_indices,
                            tf.cast(tf.fill([batch_size], t - 1), dtype=tf.int64),
                            tf.cast(tf.fill([batch_size], u - 1), dtype=tf.int64),
                            label_indices,
                        ],
                        axis=1,
                    )

                    p_label = tf.gather_nd(log_probs, indices_to_gather)

                    # Read alpha from the current time step but previous label position
                    log_alpha_label = log_alpha_from_left_prev + p_label

                    # Combine paths using log-sum-exp for stability
                    current_alpha = tf.math.reduce_logsumexp(
                        tf.stack([log_alpha_blank, log_alpha_label]), axis=0
                    )
                    inner_ta = inner_ta.write(u, current_alpha)

            # Write the newly computed alpha row for time `t`
            stacked_inner = inner_ta.stack()
            alpha_ta = alpha_ta.write(t, tf.transpose(stacked_inner, perm=[1, 0]))

        final_alpha_TBU = alpha_ta.stack()
        return tf.transpose(final_alpha_TBU, perm=[1, 0, 2])

    # Run the computation
    alpha = compute_alpha()

    # --- Final Loss Calculation ---

    # Create indices to gather the final alpha values for each sequence
    # The final state is at (logit_length - 1, label_length)
    batch_indices = tf.range(batch_size, dtype=tf.int32)
    final_state_indices = tf.stack([batch_indices, logit_length, label_length], axis=1)

    # Gather the log-probabilities of the final states from alpha
    final_log_probs = tf.gather_nd(alpha, final_state_indices)

    # Return the negative log-likelihood as the loss
    return -final_log_probs
