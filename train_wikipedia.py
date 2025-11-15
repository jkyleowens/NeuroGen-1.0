#!/usr/bin/env python3
"""
Main Training Script - Train Autonomous Agent on Wikipedia Data

This script:
1. Downloads Wikipedia articles
2. Trains a SentencePiece tokenizer on Wikipedia text
3. Trains the modular autonomous agent on Wikipedia content
4. Saves checkpoints and monitors progress

Usage:
    python train_wikipedia.py --num_articles 1000 --epochs 5
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import random
import subprocess
import sys
import signal
import sentencepiece as spm
import threading
import queue
from sentencepiece_module import TokenizerModule


class WikipediaDataLoader:
    """Download and preprocess Wikipedia articles"""
    
    def __init__(self, cache_dir='./wikipedia_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Bot)'
        })
        
    def get_random_articles(self, num_articles=100):
        """
        Get random Wikipedia articles.
        
        Args:
            num_articles: Number of articles to fetch
            
        Returns:
            List of article dictionaries
        """
        print(f"\n[Wikipedia] Fetching {num_articles} random articles...")
        
        articles = []
        batch_size = 10
        
        for i in tqdm(range(0, num_articles, batch_size)):
            try:
                # Use Wikipedia API to get random pages
                url = "https://en.wikipedia.org/w/api.php"
                params = {
                    'action': 'query',
                    'format': 'json',
                    'list': 'random',
                    'rnlimit': min(batch_size, num_articles - i),
                    'rnnamespace': 0  # Main namespace only
                }
                
                response = self.session.get(url, params=params, timeout=10)
                data = response.json()
                
                # Get content for each page
                for page in data['query']['random']:
                    article = self._get_article_content(page['id'], page['title'])
                    if article and len(article['text']) > 200:  # Filter short articles
                        articles.append(article)
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"\n[Warning] Error fetching batch: {str(e)}")
                continue
        
        print(f"[Wikipedia] Successfully fetched {len(articles)} articles")
        return articles
    
    def _get_article_content(self, page_id, title):
        """Get full content of a Wikipedia article"""
        try:
            url = "https://en.wikipedia.org/w/api.php"
            params = {
                'action': 'query',
                'format': 'json',
                'pageids': page_id,
                'prop': 'extracts',
                'explaintext': True,
                'exsectionformat': 'plain'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            data = response.json()
            
            pages = data['query']['pages']
            page_data = pages[str(page_id)]
            
            if 'extract' in page_data:
                text = page_data['extract']
                
                # Clean text
                text = text.strip()
                
                # Get summary (first paragraph)
                paragraphs = text.split('\n\n')
                summary = paragraphs[0] if paragraphs else text[:500]
                
                return {
                    'id': page_id,
                    'title': title,
                    'url': f"https://en.wikipedia.org/?curid={page_id}",
                    'text': text,
                    'summary': summary,
                    'length': len(text)
                }
            
        except Exception as e:
            print(f"\n[Warning] Error fetching article {title}: {str(e)}")
        
        return None
    
    def save_articles(self, articles, filename='articles.json'):
        """Save articles to cache"""
        filepath = self.cache_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        print(f"[Wikipedia] Saved {len(articles)} articles to {filepath}")
    
    def load_articles(self, filename='articles.json'):
        """Load articles from cache"""
        filepath = self.cache_dir / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            print(f"[Wikipedia] Loaded {len(articles)} articles from cache")
            return articles
        return None
    
    def create_tokenizer_corpus(self, articles, output_file='wikipedia_corpus.txt'):
        """
        Create a text corpus for tokenizer training.
        
        Args:
            articles: List of article dictionaries
            output_file: Output filename
            
        Returns:
            Path to corpus file
        """
        print(f"\n[Wikipedia] Creating tokenizer corpus...")
        
        corpus_path = self.cache_dir / output_file
        
        with open(corpus_path, 'w', encoding='utf-8') as f:
            for article in tqdm(articles):
                # Write title
                f.write(article['title'] + '\n')
                
                # Write text, one sentence per line (approximate)
                text = article['text']
                # Simple sentence splitting
                sentences = text.replace('? ', '?\n').replace('! ', '!\n').replace('. ', '.\n')
                f.write(sentences + '\n\n')
        
        # Get file size
        size_mb = corpus_path.stat().st_size / (1024 * 1024)
        print(f"[Wikipedia] Corpus created: {corpus_path} ({size_mb:.2f} MB)")
        
        return str(corpus_path)


class NeuroGenAgent:
    """Interface to the C++ NeuroGen autonomous agent"""
    
    def __init__(self, executable_path='./NeuroGen', config=None, tokenizer=None):
        """
        Initialize the NeuroGen agent interface.
        
        Args:
            executable_path: Path to NeuroGen executable
            config: Agent configuration
            tokenizer: TokenizerModule instance for text processing
        """
        self.executable_path = executable_path
        self.config = config or {}
        self.tokenizer = tokenizer
        self.process = None
        self.stdout_queue = queue.Queue()
        self.stderr_queue = queue.Queue()
        self.stats = {
            'browsing': {'urls_visited': 0, 'pages_read': 0},
            'learning': {'total_reward': 0.0, 'curiosity_score': 0.5},
            'memory': {'working_memory_size': 0, 'episodic_memory_size': 0, 'total_stored': 0},
            'modules': {},
            'tokenizer': {'vocab_size': 0, 'tokens_processed': 0}
        }
        self.curiosity_score = 0.5
        
    def _enqueue_output(self, out, q):
        """Thread target to read a stream and put lines into a queue."""
        for line in iter(out.readline, b''):
            q.put(line)
        out.close()

    def _read_streams(self):
        """Read all available lines from stdout and stderr queues."""
        stdout_lines = []
        stderr_lines = []
        while not self.stdout_queue.empty():
            try:
                stdout_lines.append(self.stdout_queue.get_nowait().decode('utf-8', errors='replace'))
            except queue.Empty:
                break
        while not self.stderr_queue.empty():
            try:
                stderr_lines.append(self.stderr_queue.get_nowait().decode('utf-8', errors='replace'))
            except queue.Empty:
                break
        return "".join(stdout_lines), "".join(stderr_lines)

    def initialize(self, corpus_path=None):
        """Initialize the NeuroGen C++ agent"""
        print(f"\n[NeuroGen] Starting C++ agent: {self.executable_path}")
        
        if not Path(self.executable_path).exists():
            raise FileNotFoundError(f"NeuroGen executable not found: {self.executable_path}")
        
        # Update tokenizer stats if available
        if self.tokenizer:
            self.stats['tokenizer']['vocab_size'] = self.tokenizer.get_vocab_size()
            print(f"[NeuroGen] Using tokenizer with vocab size: {self.stats['tokenizer']['vocab_size']}")
        
        # Start the NeuroGen process
        try:
            self.process = subprocess.Popen(
                [self.executable_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Start threads to read stdout and stderr
            self.stdout_thread = threading.Thread(target=self._enqueue_output, args=(self.process.stdout, self.stdout_queue))
            self.stderr_thread = threading.Thread(target=self._enqueue_output, args=(self.process.stderr, self.stderr_queue))
            self.stdout_thread.daemon = True
            self.stderr_thread.daemon = True
            self.stdout_thread.start()
            self.stderr_thread.start()

            # Wait for initialization
            time.sleep(2)
            
            stdout, stderr = self._read_streams()
            if stdout:
                print(f"[NeuroGen STDOUT] {stdout.strip()}")
            if stderr:
                print(f"[NeuroGen STDERR] {stderr.strip()}")

            if self.process.poll() is not None:
                raise RuntimeError(f"NeuroGen process failed to start. Stderr: {stderr}")
            
            print("[NeuroGen] C++ agent initialized successfully")
            return True
            
        except Exception as e:
            print(f"[Error] Failed to initialize NeuroGen: {e}")
            return False
    
    def browse_page(self, url, content):
        """
        Send article content to the NeuroGen agent for processing.
        
        Args:
            url: Article URL
            content: Article text content
            
        Returns:
            Result dictionary with reward and token info
        """
        if not self.process or self.process.poll() is not None:
            print("[Warning] NeuroGen process not running")
            return {'reward': 0.0, 'error': 'Process not running', 'tokens': []}
        
        try:
            # Tokenize content if tokenizer is available
            tokens = []
            tokenized_text = content[:2000]
            
            if self.tokenizer:
                try:
                    # Encode text to token IDs
                    tokens = self.tokenizer.encode(content[:2000], add_bos=True, add_eos=True)
                    self.stats['tokenizer']['tokens_processed'] += len(tokens)
                    
                    # Convert tokens to string representation for processing
                    token_pieces = self.tokenizer.encode_as_pieces(content[:2000])
                    tokenized_text = " ".join(token_pieces[:100])  # Limit to first 100 pieces
                    
                    if len(tokens) > 0:
                        print(f"[Tokenizer] Encoded {len(tokens)} tokens from text")
                except Exception as e:
                    print(f"[Warning] Tokenization failed: {e}")
                    tokenized_text = content[:2000]
            
            # Send content to NeuroGen via stdin using the command format
            command = f"process_text: {tokenized_text}\n"
            content_bytes = command.encode('utf-8', errors='replace')
            self.process.stdin.write(content_bytes)
            self.process.stdin.flush()

            # Give the process a moment to respond
            time.sleep(0.1)

            # Read any output to prevent buffer overflow
            stdout, stderr = self._read_streams()
            if stdout:
                print(f"[NeuroGen STDOUT] {stdout.strip()}")
            if stderr:
                print(f"[NeuroGen STDERR] {stderr.strip()}")

            # Enhanced reward mechanism considering tokenization
            base_reward = len(stdout.strip()) / 100.0
            token_reward = len(tokens) / 1000.0 if tokens else 0.0
            reward = base_reward + token_reward
            
            # Update stats
            self.stats['browsing']['urls_visited'] += 1
            self.stats['browsing']['pages_read'] += 1
            
            return {'reward': reward, 'tokens': tokens, 'num_tokens': len(tokens)}
            
        except (IOError, BrokenPipeError) as e:
            print(f"[Error] Communication error with NeuroGen process: {e}")
            self.shutdown()
            return {'reward': 0.0, 'error': str(e), 'tokens': []}
        except Exception as e:
            print(f"[Error] An unexpected error occurred in browse_page: {e}")
            return {'reward': 0.0, 'error': str(e), 'tokens': []}
    
    def save(self, path):
        """Save agent state (C++ handles this internally)"""
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Python-side stats
        stats_file = save_dir / 'python_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        # Save tokenizer state if available
        if self.tokenizer:
            tokenizer_dir = save_dir / 'tokenizer'
            self.tokenizer.save_state(str(tokenizer_dir))
            print(f"[NeuroGen] Saved tokenizer state")
        
        print(f"[NeuroGen] Saved state to {path}")
    
    def get_statistics(self):
        """Get current statistics"""
        return self.stats
    
    def search_memories(self, query):
        """Search memories (stub for compatibility)"""
        return {
            'memories': [],
            'query': query
        }
    
    def shutdown(self):
        """Shutdown the NeuroGen agent"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                print("[NeuroGen] Agent shut down successfully")
            except Exception as e:
                print(f"[Warning] Error shutting down agent: {e}")
                self.process.kill()
        
        if self.process and self.process.poll() is None:
            print("\n[NeuroGen] Shutting down C++ agent...")
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                print("[NeuroGen] C++ agent shut down.")
            except (IOError, BrokenPipeError):
                # Process might already be gone
                pass
            except subprocess.TimeoutExpired:
                print("[Warning] NeuroGen process did not terminate gracefully, killing.")
                self.process.kill()
            
            # Read any final output
            stdout, stderr = self._read_streams()
            if stdout:
                print(f"[NeuroGen Final STDOUT] {stdout.strip()}")
            if stderr:
                print(f"[NeuroGen Final STDERR] {stderr.strip()}")

        self.process = None


class WikipediaTrainer:
    """Train autonomous agent on Wikipedia content"""
    
    def __init__(self, agent, articles, config=None):
        """
        Initialize trainer.
        
        Args:
            agent: NeuroGenAgent instance
            articles: List of Wikipedia articles
            config: Training configuration
        """
        self.agent = agent
        self.articles = articles
        self.config = config or {}
        
        # Training state
        self.epoch = 0
        self.article_idx = 0
        self.total_reward = 0.0
        self.episode_rewards = []
        
        # Metrics
        self.metrics = {
            'articles_processed': 0,
            'total_reward': 0.0,
            'avg_reward_per_article': 0.0,
            'learning_progress': []
        }
    
    def train(self, num_epochs=5, articles_per_epoch=None, checkpoint_every=50):
        """
        Train agent on Wikipedia articles.
        
        Args:
            num_epochs: Number of training epochs
            articles_per_epoch: Articles per epoch (None = all)
            checkpoint_every: Save checkpoint every N articles
        """
        print("\n" + "="*60)
        print("TRAINING AUTONOMOUS AGENT ON WIKIPEDIA")
        print("="*60)
        
        articles_per_epoch = articles_per_epoch or len(self.articles)
        total_articles = num_epochs * articles_per_epoch
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Articles per epoch: {articles_per_epoch}")
        print(f"  Total articles: {total_articles}")
        print(f"  Checkpoint every: {checkpoint_every} articles")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            print(f"\n{'='*60}")
            print(f"EPOCH {self.epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Shuffle articles each epoch
            epoch_articles = random.sample(self.articles, 
                                         min(articles_per_epoch, len(self.articles)))
            
            epoch_reward = 0.0
            
            # Process each article
            for i, article in enumerate(tqdm(epoch_articles, desc=f"Epoch {self.epoch}")):
                self.article_idx += 1
                
                # Process article
                reward = self._process_article(article)
                epoch_reward += reward
                self.total_reward += reward
                
                # Save checkpoint
                if self.article_idx % checkpoint_every == 0:
                    self._save_checkpoint()
                
                # Log progress
                if (i + 1) % 10 == 0:
                    avg_reward = epoch_reward / (i + 1)
                    print(f"\n  [{self.epoch}.{i+1}] Avg reward: {avg_reward:.3f}, "
                          f"Total: {self.total_reward:.2f}")
            
            # Epoch summary
            avg_epoch_reward = epoch_reward / len(epoch_articles)
            self.episode_rewards.append(avg_epoch_reward)
            
            print(f"\nEpoch {self.epoch} Summary:")
            print(f"  Articles processed: {len(epoch_articles)}")
            print(f"  Average reward: {avg_epoch_reward:.3f}")
            print(f"  Cumulative reward: {self.total_reward:.2f}")
            
            # Update metrics
            self._update_metrics()
            
            # Save after each epoch
            self._save_checkpoint(epoch_complete=True)
        
        # Training complete
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {elapsed/60:.2f} minutes")
        print(f"Articles processed: {self.article_idx}")
        print(f"Final reward: {self.total_reward:.2f}")
        print(f"Avg reward/article: {self.total_reward/self.article_idx:.3f}")
        
        # Save final model
        self._save_final_model()
        
        return self.metrics
    
    def _process_article(self, article):
        """
        Process a single Wikipedia article.
        
        Args:
            article: Article dictionary
            
        Returns:
            Reward value
        """
        # Simulate browsing the article
        result = self.agent.browse_page(
            url=article['url'],
            content=article['text']
        )
        
        # Calculate reward based on multiple factors
        reward = result['reward']
        
        # Additional reward for longer, more informative articles
        length_bonus = min(0.1, article['length'] / 10000)
        reward += length_bonus
        
        # Reward for maintaining curiosity (exploration)
        curiosity_bonus = self.agent.curiosity_score * 0.1
        reward += curiosity_bonus
        
        reward = np.clip(reward, 0, 1)
        
        return reward
    
    def _update_metrics(self):
        """Update training metrics"""
        self.metrics['articles_processed'] = self.article_idx
        self.metrics['total_reward'] = self.total_reward
        self.metrics['avg_reward_per_article'] = self.total_reward / max(1, self.article_idx)
        self.metrics['learning_progress'].append({
            'epoch': self.epoch,
            'articles': self.article_idx,
            'reward': self.total_reward,
            'avg_reward': self.metrics['avg_reward_per_article']
        })
    
    def _save_checkpoint(self, epoch_complete=False):
        """Save training checkpoint"""
        checkpoint_dir = Path('./checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        
        if epoch_complete:
            checkpoint_name = f'checkpoint_epoch_{self.epoch}'
        else:
            checkpoint_name = f'checkpoint_article_{self.article_idx}'
        
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        # Save agent
        self.agent.save(str(checkpoint_path))
        
        # Save training state
        training_state = {
            'epoch': self.epoch,
            'article_idx': self.article_idx,
            'total_reward': self.total_reward,
            'episode_rewards': self.episode_rewards,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path / 'training_state.json', 'w') as f:
            json.dump(training_state, f, indent=2)
        
        if epoch_complete:
            print(f"\n[Checkpoint] Saved epoch {self.epoch} to {checkpoint_path}")
    
    def _save_final_model(self):
        """Save final trained model"""
        final_dir = Path('./trained_agent_final')
        
        # Save agent
        self.agent.save(str(final_dir))
        
        # Save complete training history
        history = {
            'total_epochs': self.epoch,
            'total_articles': self.article_idx,
            'final_reward': self.total_reward,
            'episode_rewards': self.episode_rewards,
            'metrics': self.metrics,
            'config': self.config,
            'completed_at': datetime.now().isoformat()
        }
        
        with open(final_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n[Final Model] Saved to {final_dir}")


def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train autonomous agent on Wikipedia')
    parser.add_argument('--num_articles', type=int, default=500,
                       help='Number of Wikipedia articles to download')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--articles_per_epoch', type=int, default=None,
                       help='Articles per epoch (default: all)')
    parser.add_argument('--checkpoint_every', type=int, default=50,
                       help='Save checkpoint every N articles')
    parser.add_argument('--use_cache', action='store_true',
                       help='Use cached articles if available')
    parser.add_argument('--vocab_size', type=int, default=32000,
                       help='Tokenizer vocabulary size')
    parser.add_argument('--embedding_dim', type=int, default=512,
                       help='Embedding dimension')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("WIKIPEDIA AUTONOMOUS AGENT TRAINING PIPELINE")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Articles to download: {args.num_articles}")
    print(f"  Training epochs: {args.epochs}")
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Embedding dimension: {args.embedding_dim}")
    print(f"  Use cache: {args.use_cache}")
    
    # Step 1: Download Wikipedia articles
    print("\n" + "="*60)
    print("STEP 1: DOWNLOADING WIKIPEDIA ARTICLES")
    print("="*60)
    
    loader = WikipediaDataLoader()
    
    if args.use_cache:
        articles = loader.load_articles()
    else:
        articles = None
    
    if articles is None or len(articles) < args.num_articles:
        articles = loader.get_random_articles(args.num_articles)
        loader.save_articles(articles)
    
    print(f"\n✓ Total articles available: {len(articles)}")
    
    # Show sample articles
    print("\nSample articles:")
    for i, article in enumerate(articles[:3]):
        print(f"  {i+1}. {article['title']} ({article['length']} chars)")
    
    # Step 2: Create tokenizer corpus and train tokenizer
    print("\n" + "="*60)
    print("STEP 2: TRAINING SENTENCEPIECE TOKENIZER")
    print("="*60)
    
    # Load existing SentencePiece tokenizer model
    tokenizer_model_path = './nlp_agent_tokenizer.model'
    tokenizer = TokenizerModule()
    
    if not Path(tokenizer_model_path).exists():
        print(f"\n[Error] Tokenizer model not found: {tokenizer_model_path}")
        print(f"Please train the tokenizer first using sentencepiece_train.py")
        print(f"Or create a corpus and train with:")
        print(f"  corpus_path = loader.create_tokenizer_corpus(articles)")
        print(f"  python sentencepiece_train.py")
        sys.exit(1)
    
    print(f"\n[Tokenizer] Loading existing tokenizer: {tokenizer_model_path}")
    tokenizer.load_model(tokenizer_model_path)
    print(f"✓ Loaded tokenizer with vocab size: {tokenizer.get_vocab_size()}")
    print(f"✓ Model file: {tokenizer_model_path}")
    print(f"✓ Vocab file: ./nlp_agent_tokenizer.vocab")
    
    # Test the tokenizer
    test_text = "The autonomous agent learns from Wikipedia articles."
    test_tokens = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    test_pieces = tokenizer.encode_as_pieces(test_text)
    print(f"\n[Tokenizer] Test encoding:")
    print(f"  Text: {test_text}")
    print(f"  Tokens: {test_tokens[:10]}... ({len(test_tokens)} total)")
    print(f"  Pieces: {test_pieces[:10]}... ({len(test_pieces)} total)")
    print(f"  Decoded: {tokenizer.decode(test_tokens)}")
    
    # Step 3: Initialize agent
    print("\n" + "="*60)
    print("STEP 3: INITIALIZING AUTONOMOUS AGENT")
    print("="*60)
    
    # Agent configuration (use actual vocab size from loaded tokenizer)
    actual_vocab_size = tokenizer.get_vocab_size()
    agent_config = {
        'embedding_dim': args.embedding_dim,
        'vocab_size': actual_vocab_size,
        'tokenizer': {
            'model_type': 'bpe',
            'max_length': 512
        },
        'embedding': {
            'use_positional_encoding': True,
            'positional_encoding_type': 'learned'
        },
        'attention': {
            'num_heads': 8,
            'dropout': 0.1
        },
        'vision': {
            'image_size': 224,
            'patch_size': 16,
            'num_layers': 6
        },
        'memory': {
            'working_memory_capacity': 100,
            'episodic_memory_capacity': 10000,
            'consolidation_threshold': 0.65
        },
        'control': {
            'max_active_modules': 5,
            'use_adaptive_routing': True,
            'learning_rate': 0.01
        }
    }
    
    agent = NeuroGenAgent(executable_path='./NeuroGen', config=agent_config, tokenizer=tokenizer)
    
    print("\nInitializing agent with tokenizer...")
    if not agent.initialize(corpus_path=None):
        print("[Error] Failed to initialize agent. Exiting.")
        sys.exit(1)
    
    print("✓ Agent initialized and ready for training")
    print(f"✓ Tokenizer integrated with {tokenizer.get_vocab_size()} vocabulary size")
    
    # Step 4: Train agent
    print("\n" + "="*60)
    print("STEP 4: TRAINING AGENT ON WIKIPEDIA")
    print("="*60)
    
    trainer = WikipediaTrainer(
        agent=agent,
        articles=articles,
        config=agent_config
    )
    
    metrics = trainer.train(
        num_epochs=args.epochs,
        articles_per_epoch=args.articles_per_epoch,
        checkpoint_every=args.checkpoint_every
    )
    
    # Step 5: Display final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    
    stats = agent.get_statistics()
    
    print(f"\nBrowsing Activity:")
    print(f"  Total URLs visited: {stats['browsing']['urls_visited']}")
    print(f"  Pages read: {stats['browsing']['pages_read']}")
    
    print(f"\nLearning Metrics:")
    print(f"  Total reward: {stats['learning']['total_reward']:.2f}")
    print(f"  Average reward/article: {metrics['avg_reward_per_article']:.3f}")
    print(f"  Curiosity score: {stats['learning']['curiosity_score']:.3f}")
    
    print(f"\nMemory System:")
    print(f"  Working memory: {stats['memory']['working_memory_size']}")
    print(f"  Episodic memory: {stats['memory']['episodic_memory_size']}")
    print(f"  Total memories: {stats['memory']['total_stored']}")
    
    print(f"\nTokenizer Performance:")
    print(f"  Vocabulary size: {stats['tokenizer']['vocab_size']}")
    print(f"  Tokens processed: {stats['tokenizer']['tokens_processed']}")
    if stats['tokenizer']['tokens_processed'] > 0 and metrics['articles_processed'] > 0:
        avg_tokens = stats['tokenizer']['tokens_processed'] / metrics['articles_processed']
        print(f"  Average tokens/article: {avg_tokens:.1f}")
    
    print(f"\nModule Performance:")
    for module_id, perf in stats['modules'].items():
        if isinstance(perf, dict) and 'activation_count' in perf:
            print(f"  {module_id}:")
            print(f"    Activations: {perf['activation_count']}")
            print(f"    Success rate: {perf['performance']['success_rate']:.2%}")
    
    # Test the trained agent
    print("\n" + "="*60)
    print("TESTING TRAINED AGENT")
    print("="*60)
    
    print("\nTesting memory retrieval...")
    test_queries = [
        "What is artificial intelligence?",
        "Tell me about neural networks",
        "What did I learn about science?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        memories = agent.search_memories(query)
        if memories['memories']:
            top_memory = memories['memories'][0]
            print(f"  Top result: {top_memory['content'][:100]}...")
            print(f"  Relevance: {top_memory['relevance_score']:.3f}")
    
    print("\n" + "="*60)
    print("TRAINING PIPELINE COMPLETE")
    print("="*60)
    print(f"\n✓ Agent trained on {metrics['articles_processed']} Wikipedia articles")
    print(f"✓ Final model saved to: ./trained_agent_final")
    print(f"✓ Checkpoints saved to: ./checkpoints")
    print(f"\nThe agent is now ready for autonomous operation!")
    
    # Cleanup
    agent.shutdown()
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Training stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Error] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)