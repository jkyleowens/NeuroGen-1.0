#!/usr/bin/env python3
"""
Optimized Training Script - Proper Synchronization for Next-Token Prediction

This version eliminates the time.sleep() delays and implements proper
subprocess communication with blocking I/O and timeout handling.

Key improvements:
- Proper blocking I/O with configurable timeouts
- Batch processing of training examples
- Robust error handling and recovery
- Performance monitoring and metrics
"""

import argparse
import json
import time
import select
import threading
import queue
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
from sentencepiece_module import TokenizerModule
import logging
import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class SynchronizedSubprocess:
    """
    Handles subprocess communication with proper synchronization.
    
    This class provides:
    - Blocking I/O with configurable timeouts
    - Proper response synchronization
    - Error handling and recovery
    - Optional batching support
    """
    
    def __init__(self, executable_path, timeout=5.0):
        self.executable_path = executable_path
        self.timeout = timeout
        self.process = None
        self.response_queue = queue.Queue()
        self.output_thread = None
        self.running = False
        
    def start(self):
        """Start the subprocess with proper I/O setup"""
        if not Path(self.executable_path).exists():
            raise FileNotFoundError(f"Executable not found: {self.executable_path}")
        
        try:
            self.process = subprocess.Popen(
                [self.executable_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=0  # Unbuffered for immediate response
            )
            
            # Start output monitoring thread
            self.running = True
            self.output_thread = threading.Thread(target=self._monitor_output, daemon=True)
            self.output_thread.start()
            
            # Wait for initial startup (with timeout)
            try:
                initial_response = self.response_queue.get(timeout=self.timeout)
                print(f"[Subprocess] Started successfully: {initial_response.strip()}")
                return True
            except queue.Empty:
                print(f"[Warning] Subprocess started but no initial response within {self.timeout}s")
                return True
                
        except Exception as e:
            print(f"[Error] Failed to start subprocess: {e}")
            return False
    
    def _monitor_output(self):
        """Monitor subprocess output in a separate thread"""
        buffer = ""
        
        while self.running and self.process and self.process.poll() is None:
            try:
                # Use select for non-blocking check with timeout
                if select.select([self.process.stdout], [], [], 0.1)[0]:
                    char = self.process.stdout.read(1)
                    if char:
                        buffer += char
                        
                        # Check for complete lines or specific markers
                        if char == '\n' or 'NEXT_WORD_PREDICTION:' in buffer:
                            self.response_queue.put(buffer)
                            buffer = ""
                    else:
                        # EOF reached
                        break
            except Exception as e:
                print(f"[Error] Output monitoring failed: {e}")
                break
    
    def send_and_wait(self, command, timeout=None):
        """
        Send a command and wait for response with proper synchronization.
        
        Args:
            command: Command string to send
            timeout: Maximum time to wait for response (uses default if None)
            
        Returns:
            Response string from subprocess
            
        Raises:
            TimeoutError: If no response within timeout
            BrokenPipeError: If subprocess communication failed
        """
        if not self.process or self.process.poll() is not None:
            raise BrokenPipeError("Subprocess is not running")
        
        timeout = timeout or self.timeout
        
        try:
            # Clear any old responses in the queue
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Send command
            self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            
            # Wait for response with timeout
            try:
                response = self.response_queue.get(timeout=timeout)
                return response.strip()
            except queue.Empty:
                raise TimeoutError(f"No response within {timeout} seconds")
                
        except BrokenPipeError:
            raise BrokenPipeError("Failed to communicate with subprocess")
    
    def send_batch_and_wait(self, commands, timeout=None):
        """
        Send multiple commands as a batch and wait for all responses.
        
        This is more efficient than individual send_and_wait calls.
        """
        if not self.process or self.process.poll() is not None:
            raise BrokenPipeError("Subprocess is not running")
        
        timeout = timeout or self.timeout * len(commands)
        responses = []
        
        try:
            # Clear old responses
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Send all commands
            for command in commands:
                self.process.stdin.write(command + '\n')
            self.process.stdin.flush()
            
            # Collect all responses
            for i in range(len(commands)):
                try:
                    response = self.response_queue.get(timeout=timeout)
                    responses.append(response.strip())
                except queue.Empty:
                    raise TimeoutError(f"Missing response {i+1}/{len(commands)} within {timeout}s")
            
            return responses
            
        except BrokenPipeError:
            raise BrokenPipeError("Failed to communicate with subprocess")
    
    def stop(self):
        """Stop the subprocess gracefully"""
        self.running = False
        
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
                print("[Subprocess] Stopped gracefully")
            except subprocess.TimeoutExpired:
                print("[Warning] Subprocess didn't terminate, killing")
                self.process.kill()
            except Exception as e:
                print(f"[Error] Error stopping subprocess: {e}")
        
        self.process = None


class OptimizedNeuroGenAgent:
    """
    Optimized NeuroGen agent with proper synchronization and batching support.
    """
    
    def __init__(self, executable_path='./neurogen', tokenizer=None):
        self.tokenizer = tokenizer
        self.subprocess = SynchronizedSubprocess(executable_path, timeout=10.0)
        self.stats = {
            'predictions': 0,
            'correct_predictions': 0,
            'total_processing_time': 0.0,
            'avg_response_time': 0.0,
            'timeouts': 0,
            'communication_errors': 0
        }
        
    def initialize(self):
        """Initialize the agent with proper startup synchronization"""
        print("[Agent] Initializing optimized NeuroGen agent...")
        
        if not self.tokenizer:
            print("[Warning] No tokenizer provided")
            return False
        
        success = self.subprocess.start()
        if success:
            print(f"[Agent] Initialized with vocab size: {self.tokenizer.get_vocab_size()}")
        
        return success
    
    def train_next_token_prediction_batch(self, url, content, batch_size=10, max_examples=50):
        """
        Optimized next-token prediction training with batching.
        
        This version processes multiple training examples in batches to
        reduce I/O overhead and improve performance.
        """
        if not self.tokenizer:
            return {'reward': 0.0, 'error': 'No tokenizer', 'tokens': []}
        
        try:
            # Tokenize content
            tokens = self.tokenizer.encode(content[:4000], add_bos=True, add_eos=True)
            
            if len(tokens) < 2:
                return {'reward': 0.0, 'error': 'Text too short', 'tokens': tokens}
            
            # Sample training positions
            num_examples = min(max_examples, len(tokens) - 1)
            training_positions = np.random.choice(
                range(1, len(tokens)),
                size=num_examples,
                replace=False
            )
            training_positions = sorted(training_positions)
            
            print(f"[Training] Processing {len(training_positions)} examples in batches of {batch_size}")
            
            total_reward = 0.0
            correct_predictions = 0
            total_examples = 0
            
            # Process in batches
            for i in range(0, len(training_positions), batch_size):
                batch_positions = training_positions[i:i + batch_size]
                batch_commands = []
                batch_targets = []
                
                # Prepare batch
                for position in batch_positions:
                    context_tokens = tokens[:position]
                    target_token = tokens[position]
                    
                    context_text = self.tokenizer.decode(context_tokens)
                    target_text = self.tokenizer.decode([target_token])
                    
                    batch_commands.append(f"process_text: {context_text}")
                    batch_targets.append(target_text)
                
                # Process batch with timeout
                try:
                    start_time = time.time()
                    responses = self.subprocess.send_batch_and_wait(
                        batch_commands, 
                        timeout=2.0 * len(batch_commands)  # Scale timeout with batch size
                    )
                    processing_time = time.time() - start_time
                    
                    # Evaluate predictions
                    batch_reward, batch_correct = self._evaluate_batch_predictions(
                        responses, batch_targets
                    )
                    
                    total_reward += batch_reward
                    correct_predictions += batch_correct
                    total_examples += len(batch_commands)
                    
                    # Update stats
                    self.stats['predictions'] += len(batch_commands)
                    self.stats['correct_predictions'] += batch_correct
                    self.stats['total_processing_time'] += processing_time
                    self.stats['avg_response_time'] = (
                        self.stats['total_processing_time'] / self.stats['predictions']
                    )
                    
                    print(f"[Batch {i//batch_size + 1}] "
                          f"{batch_correct}/{len(batch_commands)} correct "
                          f"({batch_correct/len(batch_commands):.1%} accuracy) "
                          f"in {processing_time:.2f}s")
                
                except TimeoutError:
                    print(f"[Warning] Batch {i//batch_size + 1} timed out")
                    self.stats['timeouts'] += 1
                    continue
                    
                except BrokenPipeError:
                    print("[Error] Communication failed, attempting recovery...")
                    self.stats['communication_errors'] += 1
                    if not self._recover_communication():
                        break
            
            # Calculate final metrics
            accuracy = correct_predictions / max(1, total_examples)
            avg_reward = total_reward / max(1, total_examples)
            
            return {
                'reward': avg_reward,
                'accuracy': accuracy,
                'correct_predictions': correct_predictions,
                'num_training_examples': total_examples,
                'tokens': tokens,
                'processing_stats': {
                    'total_time': self.stats['total_processing_time'],
                    'avg_response_time': self.stats['avg_response_time'],
                    'timeouts': self.stats['timeouts'],
                    'communication_errors': self.stats['communication_errors']
                }
            }
            
        except Exception as e:
            print(f"[Error] Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'reward': 0.0, 'error': str(e), 'tokens': []}
    
    def _evaluate_batch_predictions(self, responses, targets):
        """Evaluate a batch of predictions against targets"""
        total_reward = 0.0
        correct_count = 0
        
        for response, target in zip(responses, targets):
            # Extract prediction from response
            predicted_text = self._extract_prediction(response)
            
            if predicted_text:
                if predicted_text.lower().strip() == target.lower().strip():
                    reward = 1.0
                    correct_count += 1
                elif target.lower() in predicted_text.lower():
                    reward = 0.5
                else:
                    reward = 0.1
            else:
                reward = 0.0
            
            total_reward += reward
        
        return total_reward, correct_count
    
    def _extract_prediction(self, response):
        """Extract prediction from subprocess response"""
        if "NEXT_WORD_PREDICTION:" in response:
            try:
                pred_line = [line for line in response.split('\n') 
                           if 'NEXT_WORD_PREDICTION:' in line][0]
                return pred_line.split('NEXT_WORD_PREDICTION:')[1].strip()
            except (IndexError, AttributeError):
                return None
        return None
    
    def _recover_communication(self):
        """Attempt to recover from communication failure"""
        print("[Recovery] Attempting to restart subprocess...")
        
        self.subprocess.stop()
        time.sleep(1)  # Brief pause for cleanup
        
        return self.subprocess.start()
    
    def get_statistics(self):
        """Get training statistics"""
        stats = self.stats.copy()
        if stats['predictions'] > 0:
            stats['overall_accuracy'] = stats['correct_predictions'] / stats['predictions']
        else:
            stats['overall_accuracy'] = 0.0
        return stats
    
    def shutdown(self):
        """Shutdown the agent"""
        self.subprocess.stop()
        print(f"[Agent] Final stats: {self.get_statistics()}")


# Update the WikipediaTrainer to use the optimized agent
class OptimizedWikipediaTrainer:
    """Optimized Wikipedia trainer with proper synchronization"""
    
    def __init__(self, agent, articles, config=None):
        self.agent = agent
        self.articles = articles
        self.config = config or {}
        
        # Training parameters
        self.batch_size = self.config.get('batch_size', 10)
        self.max_examples_per_article = self.config.get('max_examples_per_article', 50)
        
        # Metrics
        self.training_start_time = None
        self.articles_processed = 0
        self.total_reward = 0.0
        self.training_history = []
        
    def train(self, num_epochs=1, articles_per_epoch=100):
        """Run optimized training loop"""
        self.training_start_time = time.time()
        
        print("="*60)
        print("🚀 STARTING OPTIMIZED TRAINING SESSION")
        print("="*60)
        print(f"Configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Articles per epoch: {articles_per_epoch}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Max examples per article: {self.max_examples_per_article}")
        print()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*40}")
            print(f"EPOCH {epoch}/{num_epochs}")
            print(f"{'='*40}")
            
            epoch_start = time.time()
            epoch_reward = 0.0
            epoch_articles = self.articles[:articles_per_epoch]
            
            for i, article in enumerate(tqdm(epoch_articles, desc=f"Epoch {epoch}")):
                try:
                    # Train with optimized batching
                    result = self.agent.train_next_token_prediction_batch(
                        url=article['url'],
                        content=article['text'],
                        batch_size=self.batch_size,
                        max_examples=self.max_examples_per_article
                    )
                    
                    reward = result.get('reward', 0.0)
                    accuracy = result.get('accuracy', 0.0)
                    num_examples = result.get('num_training_examples', 0)
                    
                    # Add bonuses
                    accuracy_bonus = accuracy * 0.2
                    length_bonus = min(0.1, article['length'] / 10000)
                    total_reward = reward + accuracy_bonus + length_bonus
                    
                    epoch_reward += total_reward
                    self.total_reward += total_reward
                    self.articles_processed += 1
                    
                    # Log progress
                    if (i + 1) % 10 == 0:
                        avg_reward = epoch_reward / (i + 1)
                        stats = self.agent.get_statistics()
                        print(f"\n[Progress] Article {i+1}/{len(epoch_articles)}")
                        print(f"  Current accuracy: {accuracy:.1%}")
                        print(f"  Average reward: {avg_reward:.3f}")
                        print(f"  Overall accuracy: {stats.get('overall_accuracy', 0):.1%}")
                        print(f"  Avg response time: {stats.get('avg_response_time', 0):.3f}s")
                        
                        if stats.get('timeouts', 0) > 0:
                            print(f"  Timeouts: {stats['timeouts']}")
                        if stats.get('communication_errors', 0) > 0:
                            print(f"  Comm errors: {stats['communication_errors']}")
                
                except Exception as e:
                    print(f"[Error] Failed to process article {i}: {e}")
                    continue
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_epoch_reward = epoch_reward / len(epoch_articles)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Time: {epoch_time/60:.2f} minutes")
            print(f"  Articles: {len(epoch_articles)}")
            print(f"  Average reward: {avg_epoch_reward:.3f}")
            
            self.training_history.append({
                'epoch': epoch,
                'reward': avg_epoch_reward,
                'time': epoch_time,
                'articles': len(epoch_articles)
            })
        
        # Final summary
        total_time = time.time() - self.training_start_time
        final_stats = self.agent.get_statistics()
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Articles processed: {self.articles_processed}")
        print(f"Total reward: {self.total_reward:.2f}")
        print(f"Average reward: {self.total_reward/max(1, self.articles_processed):.3f}")
        print(f"Final accuracy: {final_stats.get('overall_accuracy', 0):.1%}")
        print(f"Average response time: {final_stats.get('avg_response_time', 0):.3f}s")
        print(f"Performance issues:")
        print(f"  Timeouts: {final_stats.get('timeouts', 0)}")
        print(f"  Communication errors: {final_stats.get('communication_errors', 0)}")
        
        return {
            'total_time': total_time,
            'articles_processed': self.articles_processed,
            'total_reward': self.total_reward,
            'training_history': self.training_history,
            'final_stats': final_stats
        }


def main():
    """Main training function with optimized implementation"""
    parser = argparse.ArgumentParser(description='Optimized Next-Token Prediction Training')
    parser.add_argument('--num_articles', type=int, default=100, 
                       help='Number of articles to use for training')
    parser.add_argument('--epochs', type=int, default=1, 
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for training examples')
    parser.add_argument('--max_examples', type=int, default=50,
                       help='Maximum training examples per article')
    parser.add_argument('--executable', type=str, default='./neurogen',
                       help='Path to NeuroGen executable')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = TokenizerModule('wikipedia_tokenizer.model')
        print(f"Tokenizer loaded with vocab size: {tokenizer.get_vocab_size()}")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    
    # Load Wikipedia articles (simplified for example)
    print(f"Loading {args.num_articles} Wikipedia articles...")
    # This would load your articles - placeholder implementation
    articles = []
    for i in range(args.num_articles):
        articles.append({
            'title': f'Sample Article {i+1}',
            'url': f'https://en.wikipedia.org/wiki/Sample_{i+1}',
            'text': f'This is sample article content {i+1} ' * 50,
            'length': 1000
        })
    
    # Initialize optimized agent
    config = {
        'batch_size': args.batch_size,
        'max_examples_per_article': args.max_examples
    }
    
    agent = OptimizedNeuroGenAgent(args.executable, tokenizer)
    
    if not agent.initialize():
        print("Failed to initialize agent")
        return
    
    # Create trainer and run
    trainer = OptimizedWikipediaTrainer(agent, articles, config)
    
    try:
        results = trainer.train(
            num_epochs=args.epochs,
            articles_per_epoch=args.num_articles
        )
        
        # Save results
        results_path = Path('training_results_optimized.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to {results_path}")
        
    finally:
        agent.shutdown()


if __name__ == '__main__':
    main()