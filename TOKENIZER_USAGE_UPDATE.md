# Tokenizer Usage Update - November 15, 2025

## Changes Made

Updated `train_wikipedia.py` to **always use the existing pre-trained SentencePiece model** instead of attempting to train a new one each time.

## Key Modifications

### Before
```python
if Path(tokenizer_model_path).exists() and args.use_cache:
    # Load existing
    tokenizer.load_model(tokenizer_model_path)
else:
    # Train new tokenizer
    spm.SentencePieceTrainer.train(...)
```

### After
```python
if not Path(tokenizer_model_path).exists():
    print("[Error] Tokenizer model not found")
    print("Please train the tokenizer first using sentencepiece_train.py")
    sys.exit(1)

# Always load existing model
tokenizer.load_model(tokenizer_model_path)
```

## Benefits

1. ✅ **Faster Startup** - No tokenizer training overhead (saves several minutes)
2. ✅ **Consistency** - Same vocabulary across all training runs
3. ✅ **Cleaner Code** - Single responsibility (training script trains, this script uses)
4. ✅ **Explicit Dependencies** - Clear that tokenizer must be pre-trained

## Workflow

### One-Time Setup
Train the tokenizer once (only needs to be done once):
```bash
python sentencepiece_train.py
```

This creates:
- `nlp_agent_tokenizer.model` (32,000 token vocabulary)
- `nlp_agent_tokenizer.vocab` (vocabulary file)

### Regular Training Runs
Just run the training script:
```bash
# No special flags needed - automatically uses existing model
python train_wikipedia.py --num_articles 500 --epochs 3
```

## What Happens Now

1. **Script starts** → Checks for `nlp_agent_tokenizer.model`
2. **If model exists** → Loads it immediately and continues
3. **If model missing** → Exits with helpful error message explaining how to train it
4. **Agent training** → Uses the loaded tokenizer throughout

## Error Handling

If you run the script without a trained tokenizer:
```
[Error] Tokenizer model not found: ./nlp_agent_tokenizer.model
Please train the tokenizer first using sentencepiece_train.py
Or create a corpus and train with:
  corpus_path = loader.create_tokenizer_corpus(articles)
  python sentencepiece_train.py
```

## Additional Changes

- **Vocab Size**: Now uses actual vocab size from loaded model instead of `--vocab_size` argument
- **Corpus Path**: No longer creates corpus during training (only needed during initial tokenizer training)
- **Initialization**: Agent initializes with `corpus_path=None` since tokenizer is already loaded

## When to Retrain the Tokenizer

You only need to retrain `nlp_agent_tokenizer.model` if:
- You want a different vocabulary size
- You're working with a significantly different text domain
- You need to add new special tokens
- The model file is corrupted or missing

## Files Modified

- `train_wikipedia.py` - Removed tokenizer training logic
- `TOKENIZER_INTEGRATION_CHANGES.md` - Updated usage instructions

## Files Required

- `nlp_agent_tokenizer.model` ✅ (already exists in repo)
- `nlp_agent_tokenizer.vocab` ✅ (already exists in repo)
- `sentencepiece_train.py` ✅ (for future retraining if needed)
- `sentencepiece_module.py` ✅ (TokenizerModule wrapper)

## Testing

To verify the changes work:
```bash
python train_wikipedia.py --num_articles 10 --epochs 1
```

You should see:
```
STEP 2: TRAINING SENTENCEPIECE TOKENIZER
========================================

[Tokenizer] Loading existing tokenizer: ./nlp_agent_tokenizer.model
✓ Loaded tokenizer with vocab size: 32000
✓ Model file: ./nlp_agent_tokenizer.model
✓ Vocab file: ./nlp_agent_tokenizer.vocab

[Tokenizer] Test encoding:
  Text: The autonomous agent learns from Wikipedia articles.
  Tokens: [2, 415, ...]... (14 total)
  ...
```
