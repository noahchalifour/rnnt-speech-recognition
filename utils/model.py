"""
Utility functions for handling model hyperparameters.

This module provides functionality to save and load model hyperparameters
to/from JSON files, facilitating model configuration persistence and reuse
across training and inference.
"""

import os
import json


def load_hparams(model_dir):
    """
    Load model hyperparameters from a JSON file in the specified directory.
    
    Args:
        model_dir (str): Path to the directory containing the hyperparameters file.
            The file is expected to be named 'hparams.json'.
            
    Returns:
        dict: A dictionary containing the model hyperparameters.
        
    Raises:
        FileNotFoundError: If the hyperparameters file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(os.path.join(model_dir, "hparams.json"), "r") as f:
        return json.load(f)


def save_hparams(hparams, model_dir):
    """
    Save model hyperparameters to a JSON file in the specified directory.
    
    Args:
        hparams (dict): Dictionary containing the model hyperparameters to save.
        model_dir (str): Path to the directory where the hyperparameters file
            will be saved as 'hparams.json'.
            
    Raises:
        FileNotFoundError: If the directory doesn't exist.
        TypeError: If the hyperparameters cannot be serialized to JSON.
    """
    with open(os.path.join(model_dir, "hparams.json"), "w") as f:
        json.dump(hparams, f)
