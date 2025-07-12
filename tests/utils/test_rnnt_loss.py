import sys
import os
import pytest
import tensorflow as tf
import numpy as np

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FILE_DIR, "..", ".."))

from utils.loss import rnnt_loss


@pytest.fixture
def test_data():
    """Fixture to generate test data for the RNN-T loss function."""
    # Define hyperparameters for the test
    batch_size = 4
    max_input_length = 15
    max_label_length = 8
    vocab_size = 12
    blank_id = 0  # Conventionally blank is 0 or vocab_size - 1

    # Generate random mock data for logits, labels, and their lengths
    logits_shape = (batch_size, max_input_length, max_label_length + 1, vocab_size)
    mock_logits = tf.random.normal(logits_shape, dtype=tf.float32)

    # Labels should not include the blank ID
    mock_labels = tf.convert_to_tensor(
        np.random.randint(1, vocab_size, size=(batch_size, max_label_length)),
        dtype=tf.int32,
    )

    # The true length of each sequence in the batch (must be <= max length)
    mock_logit_length = tf.convert_to_tensor(
        np.random.randint(
            max_input_length // 2, max_input_length + 1, size=(batch_size,)
        ),
        dtype=tf.int32,
    )
    mock_label_length = tf.convert_to_tensor(
        np.random.randint(
            max_label_length // 2, max_label_length + 1, size=(batch_size,)
        ),
        dtype=tf.int32,
    )

    return {
        "logits": mock_logits,
        "labels": mock_labels,
        "logit_length": mock_logit_length,
        "label_length": mock_label_length,
        "blank_id": blank_id,
        "batch_size": batch_size,
        "max_input_length": max_input_length,
        "max_label_length": max_label_length,
        "vocab_size": vocab_size,
    }


def test_rnnt_loss_shapes(test_data):
    """Test that the RNN-T loss function returns outputs with the expected shape."""
    # Calculate the loss
    loss_values = rnnt_loss(
        logits=test_data["logits"],
        labels=test_data["labels"],
        logit_length=test_data["logit_length"],
        label_length=test_data["label_length"],
        blank_id=test_data["blank_id"],
    )

    # Check that the loss tensor has the expected shape (batch_size,)
    assert loss_values.shape == (test_data["batch_size"],), (
        f"Expected loss shape {(test_data['batch_size'],)}, got {loss_values.shape}"
    )

    # Check that the loss values are finite (not NaN or Inf)
    assert np.all(np.isfinite(loss_values.numpy())), "Loss values contain NaN or Inf"


def test_rnnt_loss_values(test_data):
    """Test that the RNN-T loss function returns reasonable values."""
    # Calculate the loss
    loss_values = rnnt_loss(
        logits=test_data["logits"],
        labels=test_data["labels"],
        logit_length=test_data["logit_length"],
        label_length=test_data["label_length"],
        blank_id=test_data["blank_id"],
    )

    # The loss should be positive for each item in the batch
    assert np.all(loss_values.numpy() > 0), "Expected all loss values to be positive"

    # Calculate the mean loss
    mean_loss = tf.reduce_mean(loss_values)

    # Check that the mean loss is a scalar and has a reasonable value
    assert mean_loss.shape == (), f"Expected mean loss shape (), got {mean_loss.shape}"
    assert np.isfinite(mean_loss.numpy()), "Mean loss is not finite"


def test_rnnt_loss_gradients(test_data):
    """Test that gradients can be computed for the RNN-T loss function."""
    # Use GradientTape to track operations for automatic differentiation
    with tf.GradientTape() as tape:
        # We need to watch the logits tensor to compute gradients with respect to it
        tape.watch(test_data["logits"])

        # Calculate the loss
        loss_values = rnnt_loss(
            logits=test_data["logits"],
            labels=test_data["labels"],
            logit_length=test_data["logit_length"],
            label_length=test_data["label_length"],
            blank_id=test_data["blank_id"],
        )
        total_loss = tf.reduce_mean(loss_values)

    # Compute the gradients of the total loss with respect to the logits
    gradients = tape.gradient(total_loss, test_data["logits"])

    # Check that gradients are not None
    assert gradients is not None, (
        "Gradients are None. The loss is not connected to the inputs."
    )

    # Check that gradients have the expected shape
    assert gradients.shape == test_data["logits"].shape, (
        f"Expected gradients shape {test_data['logits'].shape}, got {gradients.shape}"
    )

    # Check that gradients do not contain NaN values
    assert not tf.reduce_any(tf.math.is_nan(gradients)), "Gradients contain NaN values"


def test_rnnt_loss_zero_length():
    """Test the RNN-T loss function with zero-length sequences."""
    batch_size = 2
    max_input_length = 10
    max_label_length = 5
    vocab_size = 10
    blank_id = 0

    # Create logits with normal values
    logits = tf.random.normal(
        (batch_size, max_input_length, max_label_length + 1, vocab_size),
        dtype=tf.float32,
    )

    # Create labels with normal values
    labels = tf.convert_to_tensor(
        np.random.randint(1, vocab_size, size=(batch_size, max_label_length)),
        dtype=tf.int32,
    )

    # First sequence has zero input length, second has normal length
    logit_length = tf.constant([0, max_input_length], dtype=tf.int32)

    # First sequence has normal label length, second has zero label length
    label_length = tf.constant([max_label_length, 0], dtype=tf.int32)

    # Calculate the loss - this should execute without errors
    loss_values = rnnt_loss(
        logits=logits,
        labels=labels,
        logit_length=logit_length,
        label_length=label_length,
        blank_id=blank_id,
    )

    # The loss should be positive for each item in the batch
    assert np.all(loss_values.numpy() > 0), "Expected all loss values to be positive"


def test_rnnt_loss_different_blank_id(test_data):
    """Test the RNN-T loss function with a different blank ID."""
    # Use a different blank ID
    new_blank_id = test_data["vocab_size"] - 1

    # Calculate the loss with the new blank ID
    loss_values_new_blank = rnnt_loss(
        logits=test_data["logits"],
        labels=test_data["labels"],
        logit_length=test_data["logit_length"],
        label_length=test_data["label_length"],
        blank_id=new_blank_id,
    )

    # Calculate the loss with the default blank ID
    loss_values_default_blank = rnnt_loss(
        logits=test_data["logits"],
        labels=test_data["labels"],
        logit_length=test_data["logit_length"],
        label_length=test_data["label_length"],
        blank_id=test_data["blank_id"],
    )

    # The loss values should be different when using a different blank ID
    assert not np.allclose(
        loss_values_new_blank.numpy(), loss_values_default_blank.numpy()
    ), "Expected different loss values when using a different blank ID"


def test_rnnt_loss_equal_lengths():
    """Test the RNN-T loss function with all sequences having the same length."""
    batch_size = 3
    input_length = 10
    label_length = 5
    vocab_size = 10
    blank_id = 0

    # Create logits with normal values
    logits = tf.random.normal(
        (batch_size, input_length, label_length + 1, vocab_size), dtype=tf.float32
    )

    # Create labels with normal values
    labels = tf.convert_to_tensor(
        np.random.randint(1, vocab_size, size=(batch_size, label_length)),
        dtype=tf.int32,
    )

    # All sequences have the same input and label length
    logit_length = tf.constant([input_length] * batch_size, dtype=tf.int32)
    label_length = tf.constant([label_length] * batch_size, dtype=tf.int32)

    # Calculate the loss
    loss_values = rnnt_loss(
        logits=logits,
        labels=labels,
        logit_length=logit_length,
        label_length=label_length,
        blank_id=blank_id,
    )

    # Check that the loss has the expected shape and values are reasonable
    assert loss_values.shape == (batch_size,), (
        f"Expected loss shape {(batch_size,)}, got {loss_values.shape}"
    )
    assert np.all(np.isfinite(loss_values.numpy())), "Loss values contain NaN or Inf"
