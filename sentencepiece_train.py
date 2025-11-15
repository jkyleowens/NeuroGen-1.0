import sentencepiece as spm

# Train a new model
spm.SentencePieceTrainer.train(
    input='wikipedia_cache/wikipedia_corpus.txt',  # Your corpus
    model_prefix='nlp_agent_tokenizer',
    vocab_size=32000,  # Adjust based on your needs
    character_coverage=0.9995,
    model_type='bpe',  # or 'unigram', 'char', 'word'
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece='<pad>',
    unk_piece='<unk>',
    bos_piece='<s>',
    eos_piece='</s>',
    user_defined_symbols=['<mask>', '<sep>', '<cls>']  # Add special tokens
)