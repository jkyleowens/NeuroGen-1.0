# SentencePiece Tokenizer Integration Changes

## Summary
Updated `train_wikipedia.py` to properly incorporate SentencePiece tokenization using the `nlp_agent_tokenizer` model.

## Key Changes

### 1. Import TokenizerModule
- Added import for `sentencepiece_module.TokenizerModule`
- This provides a clean interface to SentencePiece functionality

### 2. Enhanced NeuroGenAgent Class
- **Constructor**: Now accepts `tokenizer` parameter to integrate tokenization
- **Stats Tracking**: Added tokenizer statistics tracking:
  - `vocab_size`: Vocabulary size of the tokenizer
  - `tokens_processed`: Total number of tokens processed

### 3. Updated `browse_page()` Method
- **Tokenization**: Text is now tokenized before processing:
  - Encodes text to token IDs using `tokenizer.encode()`
  - Converts to token pieces for readable representation
  - Tracks number of tokens processed
- **Enhanced Rewards**: Reward calculation now includes token-based metrics
- **Stats Updates**: Automatically updates browsing statistics

### 4. Tokenizer State Persistence
- **Save**: Tokenizer state is saved alongside agent checkpoints
- **Load**: Can restore tokenizer from saved state

### 5. Main Training Pipeline Updates

#### Step 2: Tokenizer Loading
- **Load Existing Model**: Always loads pre-trained `nlp_agent_tokenizer.model`
- **Error Handling**: Exits with helpful message if model file not found
- **Validation**: Displays vocab size and model file locations
- **Test Encoding**: Validates tokenizer with sample text to ensure it's working

#### Step 3: Agent Initialization
- Passes tokenizer instance to `NeuroGenAgent`
- Uses actual vocabulary size from loaded tokenizer (ignores `--vocab_size` arg)
- Displays vocabulary size on initialization

#### Step 5: Statistics Display
- Shows tokenizer performance metrics:
  - Vocabulary size
  - Total tokens processed
  - Average tokens per article

## Usage

### Prerequisites
The script now requires a pre-trained tokenizer model. Train it once using:
```bash
python sentencepiece_train.py
```

This creates:
- `nlp_agent_tokenizer.model`
- `nlp_agent_tokenizer.vocab`

### Training with Existing Tokenizer
```bash
python train_wikipedia.py --num_articles 500 --epochs 3
```

The script will automatically load the existing `nlp_agent_tokenizer.model` file.

## Benefits

1. **Proper Text Encoding**: Articles are now tokenized into subword units before processing
2. **Vocabulary Control**: Fixed vocabulary size prevents unbounded token spaces
3. **Efficient Representation**: BPE tokenization balances vocabulary size and representation quality
4. **State Persistence**: Tokenizer state is saved with model checkpoints
5. **Better Metrics**: Token-level statistics provide insight into processing efficiency

## File Dependencies

- `train_wikipedia.py`: Main training script (modified)
- `sentencepiece_module.py`: TokenizerModule wrapper class
- `nlp_agent_tokenizer.model`: Trained SentencePiece model
- `nlp_agent_tokenizer.vocab`: Vocabulary file

## Integration Details

### Text Processing Flow
1. Raw text from Wikipedia article
2. Tokenize to IDs: `tokenizer.encode(text, add_bos=True, add_eos=True)`
3. Convert to pieces: `tokenizer.encode_as_pieces(text)`
4. Send tokenized representation to C++ agent
5. Track tokens processed in statistics

### Checkpoint Structure
```
checkpoints/checkpoint_epoch_N/
├── python_stats.json          # Python-side statistics
├── tokenizer/                 # Tokenizer state
│   ├── tokenizer_state.json   # Tokenizer configuration
│   └── nlp_agent_tokenizer.model  # SentencePiece model
└── training_state.json        # Training state
```

## Testing

The script includes a tokenizer test after training:
- Encodes sample text: "The autonomous agent learns from Wikipedia articles."
- Displays token IDs, pieces, and decoded text
- Validates round-trip encoding/decoding

## Notes

- Tokenizer is trained once and reused across epochs
- Token statistics accumulate across all articles
- Tokenization happens before sending to C++ agent
- Special tokens (BOS/EOS) are added for proper sequence delimiting
