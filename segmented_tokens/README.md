# Segmented Tokens

This directory contains pre-processed and tokenized audio data from the LJSpeech dataset.

## Files

- `manifest.json` - Metadata about the tokenized dataset
- `shard_*.pt` - Tokenized audio data shards (not included in repository)

## Usage

To generate this data:

```bash
python main.py prepare-dataset --data-root ./data --output-dir ./segmented_tokens
```

**Note**: The actual shard files are not included in the repository due to size constraints. Run the dataset preparation command to generate them locally.
