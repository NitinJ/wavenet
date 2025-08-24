# WaveNet Checkpoints

This directory contains model checkpoints and training progress from WaveNet training.

## Files

- `best_model.pth` - Best performing model checkpoint
- `latest_checkpoint.pth` - Most recent model checkpoint  
- `progress_epoch_*.png` - Training progress plots
- `training_summary.json` - Final training metrics (if exists)

## Usage

These checkpoints are generated during training:

```bash
python main.py train --epochs 50 --batch-size 4
```

For inference using checkpoints:

```bash
python main.py inference --checkpoint-dir ./wavenet_checkpoints --generate-seconds 5.0
```

**Note**: Actual checkpoint files (.pth) are not included in the repository due to size constraints. Training will generate them locally.
