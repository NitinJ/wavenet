#!/usr/bin/env python3
"""
Main entry point for WaveNet training, inference, and dataset preparation.

This script provides a unified interface for all WaveNet operations:
- Dataset preparation and tokenization
- Model training (full dataset or debug mode)
- Audio generation and inference
- Model evaluation and testing
"""

import argparse
import torch
import torch.optim as optim
import os
import sys
from pathlib import Path
import warnings

# Import all our modules
from config import (
    get_config, get_device, setup_torch, 
    get_checkpoint_config, get_training_config
)
from model import Wavenet, TrainableModel, EvalModel, codec
from dataset import (
    create_datasets, create_fake_datasets, 
    SegmentedTokensOnDisk, AudioProcessor,
    stream_preprocess_to_shards
)
from trainer import Trainer, sanity_check_batch, test_model_output_consistency
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")
