import sentencepiece as spm
import pickle
import json
from pathlib import Path

class TokenizerModule:
    """Self-contained tokenization module with state management"""
    
    def __init__(self, model_path=None, config=None):
        self.model_path = model_path
        self.config = config or {}
        self.sp = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def train(self, input_file, vocab_size=32000, model_type='bpe'):
        """Train tokenizer on corpus"""
        model_prefix = self.config.get('model_prefix', 'tokenizer')
        
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=0.9995,
            pad_id=0, unk_id=1, bos_id=2, eos_id=3
        )
        
        self.model_path = f"{model_prefix}.model"
        self.load_model(self.model_path)
        
    def load_model(self, model_path):
        """Load trained model"""
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.model_path = model_path
        
    def encode(self, text, add_bos=False, add_eos=False):
        """Encode text to token IDs"""
        if self.sp is None:
            raise ValueError("Model not loaded. Train or load a model first.")
        return self.sp.encode(text, add_bos=add_bos, add_eos=add_eos)
    
    def decode(self, ids):
        """Decode token IDs to text"""
        if self.sp is None:
            raise ValueError("Model not loaded")
        return self.sp.decode(ids)
    
    def encode_as_pieces(self, text):
        """Encode to subword pieces (human readable)"""
        return self.sp.encode_as_pieces(text)
    
    def get_vocab_size(self):
        """Get vocabulary size"""
        return self.sp.get_piece_size() if self.sp else 0
    
    def save_state(self, save_dir):
        """Save module state"""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        state = {
            'model_path': self.model_path,
            'config': self.config,
            'vocab_size': self.get_vocab_size()
        }
        
        with open(save_dir / 'tokenizer_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        # Copy model file if it exists
        if self.model_path and Path(self.model_path).exists():
            import shutil
            shutil.copy(self.model_path, save_dir / Path(self.model_path).name)
    
    def load_state(self, save_dir):
        """Load module state"""
        save_dir = Path(save_dir)
        
        with open(save_dir / 'tokenizer_state.json', 'r') as f:
            state = json.load(f)
        
        self.config = state['config']
        model_file = save_dir / Path(state['model_path']).name
        
        if model_file.exists():
            self.load_model(str(model_file))

