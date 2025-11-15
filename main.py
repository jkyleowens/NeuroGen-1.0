class ModularNLPAgent:
    def __init__(self):
        self.modules = {}
        
        # Initialize tokenizer module
        self.modules['tokenizer'] = TokenizerModule()
        
        # Initialize other modules
        self.modules['embedding'] = EmbeddingModule()
        self.modules['encoder'] = EncoderModule()
        self.modules['attention'] = AttentionModule()
        # ... other modules
        
    def process_text(self, text):
        # Tokenize
        token_ids = self.modules['tokenizer'].encode(text, add_bos=True, add_eos=True)
        
        # Pass to embedding module
        embeddings = self.modules['embedding'](token_ids)
        
        # Continue through pipeline...
        return embeddings
    
    def save_all_modules(self, base_dir):
        """Save all module states independently"""
        base_dir = Path(base_dir)
        for name, module in self.modules.items():
            module.save_state(base_dir / name)
    
    def load_all_modules(self, base_dir):
        """Load all module states"""
        base_dir = Path(base_dir)
        for name, module in self.modules.items():
            module.load_state(base_dir / name)