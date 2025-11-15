#!/usr/bin/env python3
"""
Main Training Script - Train Autonomous Agent on Wikipedia Data with Next-Token Prediction

This script implements proper next-token prediction training as used in modern LLMs:
1. Downloads Wikipedia articles
2. Trains a SentencePiece tokenizer on Wikipedia text
3. Trains the modular autonomous agent using next-token prediction:
   - For each text, creates multiple training examples
   - At each position i, the model sees tokens[0:i] and predicts token[i]
   - Provides immediate feedback based on prediction accuracy
4. Saves checkpoints and monitors training progress

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
import logging
import csv
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


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


class TrainingLogger:
    """Comprehensive logging and metrics tracking for training"""
    
    def __init__(self, log_dir='./training_logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for this training session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.article_logs = []
        self.token_stats = defaultdict(int)
        
        # Learning progress tracking
        self.metrics['reward_progression'] = []
        self.metrics['comprehension_scores'] = []
        self.metrics['correct_answers'] = []
        self.metrics['learning_rate'] = []
        
        # Performance tracking
        self.start_time = time.time()
        self.article_times = []
        
        logging.info(f"Training session started: {self.session_id}")
        logging.info(f"Log directory: {self.session_dir}")
    
    def setup_logging(self):
        """Setup file and console logging"""
        log_file = self.session_dir / 'training.log'
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def log_article_processing(self, article_idx, article, response, tokens_processed, reward, processing_time):
        """Log detailed information about each article processed"""
        article_log = {
            'index': article_idx,
            'title': article['title'],
            'url': article['url'],
            'text_length': len(article['text']),
            'input_text': article['text'][:1000],  # Store first 1000 chars
            'full_input_text': article['text'],  # Store full text
            'agent_response': response.get('generated_text', ''),
            'comprehension_score': response.get('comprehension', 0.0),
            'tokens_processed': tokens_processed,
            'reward': reward,
            'processing_time_ms': processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Track learning progress metrics
        if 'comprehension' in response:
            article_log['comprehension'] = response['comprehension']
            self.metrics['comprehension_scores'].append(response['comprehension'])
        
        # Extract token IDs from response if available
        if 'num_tokens' in response:
            article_log['num_tokens'] = response['num_tokens']
            self.token_stats['total_tokens'] += response['num_tokens']
            self.token_stats['articles_with_tokens'] += 1
        
        # Track token generation
        if 'tokens' in response and len(response['tokens']) > 0:
            article_log['generated_tokens'] = response['tokens']
            self.token_stats['total_generated_tokens'] += len(response['tokens'])
        
        self.article_logs.append(article_log)
        self.article_times.append(processing_time)
        
        # Track reward progression for learning curves
        self.metrics['reward_progression'].append(reward)
        
        # Log to file
        logging.info(f"Article {article_idx}: '{article['title'][:50]}...' - "
                    f"Reward: {reward:.3f}, Comprehension: {article_log.get('comprehension', 0):.3f}, "
                    f"Tokens: {tokens_processed}, Time: {processing_time:.1f}ms")
        
        # Save detailed entry to separate file
        detailed_log_file = self.session_dir / 'detailed_interactions.jsonl'
        with open(detailed_log_file, 'a') as f:
            json.dump(article_log, f)
            f.write('\n')
    
    def log_epoch_metrics(self, epoch, articles_processed, avg_reward, total_reward):
        """Log epoch-level metrics"""
        self.metrics['epoch'].append(epoch)
        self.metrics['avg_reward'].append(avg_reward)
        self.metrics['total_reward'].append(total_reward)
        self.metrics['articles_processed'].append(articles_processed)
        
        logging.info(f"Epoch {epoch} complete - Articles: {articles_processed}, "
                    f"Avg Reward: {avg_reward:.3f}, Total: {total_reward:.2f}")
    
    def log_training_step(self, step, reward, comprehension=None, tokens=None):
        """Log individual training step metrics"""
        self.metrics['step'].append(step)
        self.metrics['step_reward'].append(reward)
        
        if comprehension is not None:
            self.metrics['comprehension'].append(comprehension)
        if tokens is not None:
            self.metrics['tokens_per_step'].append(tokens)
    
    def save_detailed_logs(self):
        """Save detailed logs to CSV files"""
        # Save article-level logs
        if self.article_logs:
            csv_path = self.session_dir / 'article_logs.csv'
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.article_logs[0].keys())
                writer.writeheader()
                writer.writerows(self.article_logs)
            logging.info(f"Saved article logs to {csv_path}")
        
        # Save aggregated metrics
        metrics_path = self.session_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(dict(self.metrics), f, indent=2)
        logging.info(f"Saved metrics to {metrics_path}")
        
        # Save token statistics
        token_stats_path = self.session_dir / 'token_stats.json'
        with open(token_stats_path, 'w') as f:
            json.dump(dict(self.token_stats), f, indent=2)
        logging.info(f"Saved token statistics to {token_stats_path}")
    
    def generate_visualizations(self):
        """Generate comprehensive training visualizations"""
        logging.info("Generating training visualizations...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Reward progression over epochs
        if 'epoch' in self.metrics and len(self.metrics['epoch']) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Average reward per epoch
            axes[0, 0].plot(self.metrics['epoch'], self.metrics['avg_reward'], 
                           marker='o', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Average Reward')
            axes[0, 0].set_title('Average Reward per Epoch')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Total reward per epoch
            axes[0, 1].plot(self.metrics['epoch'], self.metrics['total_reward'], 
                           marker='s', color='green', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Total Reward')
            axes[0, 1].set_title('Total Reward per Epoch')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Articles processed per epoch
            axes[1, 0].bar(self.metrics['epoch'], self.metrics['articles_processed'], 
                          color='steelblue', alpha=0.7)
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Articles Processed')
            axes[1, 0].set_title('Articles Processed per Epoch')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Cumulative reward
            cumulative_reward = np.cumsum(self.metrics['total_reward'])
            axes[1, 1].plot(self.metrics['epoch'], cumulative_reward, 
                           marker='d', color='orange', linewidth=2, markersize=8)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Cumulative Reward')
            axes[1, 1].set_title('Cumulative Reward Over Training')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.session_dir / 'reward_progression.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved reward progression chart")
        
        # 2. Processing time analysis
        if self.article_times:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Processing time distribution
            axes[0].hist(self.article_times, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Processing Time (ms)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Article Processing Time Distribution')
            axes[0].axvline(np.mean(self.article_times), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(self.article_times):.1f}ms')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Processing time over articles
            axes[1].plot(range(len(self.article_times)), self.article_times, 
                        alpha=0.6, linewidth=1)
            axes[1].set_xlabel('Article Index')
            axes[1].set_ylabel('Processing Time (ms)')
            axes[1].set_title('Processing Time Over Training')
            # Add moving average
            if len(self.article_times) > 10:
                moving_avg = np.convolve(self.article_times, np.ones(10)/10, mode='valid')
                axes[1].plot(range(9, len(self.article_times)), moving_avg, 
                           color='red', linewidth=2, label='Moving Average (10)')
                axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.session_dir / 'processing_time_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved processing time analysis")
        
        # 3. Token statistics
        if self.article_logs and any('num_tokens' in log for log in self.article_logs):
            token_counts = [log.get('num_tokens', 0) for log in self.article_logs if 'num_tokens' in log]
            
            if token_counts:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # Token distribution
                axes[0].hist(token_counts, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
                axes[0].set_xlabel('Tokens per Article')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title('Token Count Distribution')
                axes[0].axvline(np.mean(token_counts), color='red', linestyle='--', 
                               label=f'Mean: {np.mean(token_counts):.1f}')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Cumulative tokens
                cumulative_tokens = np.cumsum(token_counts)
                axes[1].plot(range(len(cumulative_tokens)), cumulative_tokens, 
                           linewidth=2, color='darkgreen')
                axes[1].set_xlabel('Article Index')
                axes[1].set_ylabel('Cumulative Tokens')
                axes[1].set_title('Cumulative Tokens Processed')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.session_dir / 'token_statistics.png', dpi=300, bbox_inches='tight')
                plt.close()
                logging.info("Saved token statistics")
        
        # 4. Reward vs article length correlation
        if self.article_logs:
            lengths = [log['text_length'] for log in self.article_logs]
            rewards = [log['reward'] for log in self.article_logs]
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            scatter = ax.scatter(lengths, rewards, alpha=0.5, s=50, c=range(len(lengths)), 
                               cmap='viridis', edgecolors='black', linewidth=0.5)
            ax.set_xlabel('Article Length (characters)')
            ax.set_ylabel('Reward')
            ax.set_title('Reward vs Article Length Correlation')
            ax.grid(True, alpha=0.3)
            
            # Add colorbar to show progression
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Article Index (Training Progress)', rotation=270, labelpad=20)
            
            # Add trend line
            z = np.polyfit(lengths, rewards, 1)
            p = np.poly1d(z)
            ax.plot(sorted(lengths), p(sorted(lengths)), "r--", alpha=0.8, 
                   label=f'Trend: y={z[0]:.2e}x+{z[1]:.3f}')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(self.session_dir / 'reward_length_correlation.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved reward-length correlation")
        
        # 5. Learning progress over time
        if self.metrics['reward_progression']:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Reward progression with moving average
            axes[0, 0].plot(self.metrics['reward_progression'], alpha=0.4, linewidth=1, label='Raw Reward')
            if len(self.metrics['reward_progression']) > 50:
                window = min(50, len(self.metrics['reward_progression']) // 10)
                moving_avg = np.convolve(self.metrics['reward_progression'], 
                                        np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.metrics['reward_progression'])), 
                               moving_avg, linewidth=2, color='red', label=f'Moving Avg ({window})')
            axes[0, 0].set_xlabel('Training Step (Article)')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Learning Progress: Reward Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Comprehension scores over time
            if self.metrics['comprehension_scores']:
                axes[0, 1].plot(self.metrics['comprehension_scores'], alpha=0.4, linewidth=1, 
                               color='green', label='Comprehension')
                if len(self.metrics['comprehension_scores']) > 50:
                    window = min(50, len(self.metrics['comprehension_scores']) // 10)
                    moving_avg = np.convolve(self.metrics['comprehension_scores'], 
                                            np.ones(window)/window, mode='valid')
                    axes[0, 1].plot(range(window-1, len(self.metrics['comprehension_scores'])), 
                                   moving_avg, linewidth=2, color='darkgreen', 
                                   label=f'Moving Avg ({window})')
                axes[0, 1].set_xlabel('Training Step (Article)')
                axes[0, 1].set_ylabel('Comprehension Score')
                axes[0, 1].set_title('Learning Progress: Comprehension Over Time')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate (reward improvement over time)
            if len(self.metrics['reward_progression']) > 100:
                window_size = 100
                learning_rate = []
                for i in range(window_size, len(self.metrics['reward_progression'])):
                    recent_avg = np.mean(self.metrics['reward_progression'][i-window_size:i])
                    previous_avg = np.mean(self.metrics['reward_progression'][i-2*window_size:i-window_size]) if i >= 2*window_size else 0
                    rate = recent_avg - previous_avg
                    learning_rate.append(rate)
                
                axes[1, 0].plot(range(window_size, len(self.metrics['reward_progression'])), 
                               learning_rate, linewidth=2, color='purple')
                axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                axes[1, 0].set_xlabel('Training Step (Article)')
                axes[1, 0].set_ylabel('Learning Rate (Δ Reward)')
                axes[1, 0].set_title('Learning Rate: Rate of Improvement')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Reward distribution over training phases
            if len(self.metrics['reward_progression']) > 300:
                phases = 4
                phase_size = len(self.metrics['reward_progression']) // phases
                phase_data = []
                phase_labels = []
                for i in range(phases):
                    start = i * phase_size
                    end = start + phase_size if i < phases - 1 else len(self.metrics['reward_progression'])
                    phase_data.append(self.metrics['reward_progression'][start:end])
                    phase_labels.append(f'Phase {i+1}\n({start}-{end})')
                
                axes[1, 1].boxplot(phase_data, labels=phase_labels)
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].set_title('Reward Distribution Across Training Phases')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(self.session_dir / 'learning_progress.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved learning progress visualization")
        
        # 6. Comprehension score distribution and statistics
        if self.metrics['comprehension_scores']:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Comprehension score histogram
            axes[0].hist(self.metrics['comprehension_scores'], bins=30, color='lightblue', 
                        edgecolor='black', alpha=0.7)
            axes[0].set_xlabel('Comprehension Score')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Comprehension Score Distribution')
            axes[0].axvline(np.mean(self.metrics['comprehension_scores']), color='red', 
                          linestyle='--', label=f'Mean: {np.mean(self.metrics["comprehension_scores"]):.3f}')
            axes[0].axvline(np.median(self.metrics['comprehension_scores']), color='green', 
                          linestyle='--', label=f'Median: {np.median(self.metrics["comprehension_scores"]):.3f}')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Comprehension improvement over training
            if len(self.metrics['comprehension_scores']) > 100:
                chunk_size = max(10, len(self.metrics['comprehension_scores']) // 20)
                chunk_means = []
                chunk_positions = []
                for i in range(0, len(self.metrics['comprehension_scores']), chunk_size):
                    chunk = self.metrics['comprehension_scores'][i:i+chunk_size]
                    chunk_means.append(np.mean(chunk))
                    chunk_positions.append(i + chunk_size // 2)
                
                axes[1].plot(chunk_positions, chunk_means, marker='o', linewidth=2, 
                           markersize=6, color='blue')
                axes[1].set_xlabel('Training Step (Article)')
                axes[1].set_ylabel('Average Comprehension Score')
                axes[1].set_title('Comprehension Improvement Over Training')
                axes[1].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(chunk_positions, chunk_means, 1)
                p = np.poly1d(z)
                axes[1].plot(chunk_positions, p(chunk_positions), "r--", alpha=0.8, 
                           label=f'Trend: slope={z[0]:.4f}')
                axes[1].legend()
            
            plt.tight_layout()
            plt.savefig(self.session_dir / 'comprehension_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            logging.info("Saved comprehension analysis")
        
        logging.info(f"All visualizations saved to {self.session_dir}")
    
    def generate_summary_report(self, training_config, final_stats):
        """Generate a comprehensive training summary report"""
        report_path = self.session_dir / 'training_summary.txt'
        
        total_time = time.time() - self.start_time
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TRAINING SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Training Time: {total_time/60:.2f} minutes\n\n")
            
            f.write("CONFIGURATION:\n")
            f.write("-"*80 + "\n")
            for key, value in training_config.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            f.write("FINAL STATISTICS:\n")
            f.write("-"*80 + "\n")
            for key, value in final_stats.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
            
            if self.article_logs:
                f.write("ARTICLE PROCESSING STATS:\n")
                f.write("-"*80 + "\n")
                f.write(f"  Total articles: {len(self.article_logs)}\n")
                
                rewards = [log['reward'] for log in self.article_logs]
                f.write(f"  Average reward: {np.mean(rewards):.4f}\n")
                f.write(f"  Reward std dev: {np.std(rewards):.4f}\n")
                f.write(f"  Min reward: {np.min(rewards):.4f}\n")
                f.write(f"  Max reward: {np.max(rewards):.4f}\n\n")
                
                times = [log['processing_time_ms'] for log in self.article_logs]
                f.write(f"  Average processing time: {np.mean(times):.1f}ms\n")
                f.write(f"  Total processing time: {sum(times)/1000:.2f}s\n\n")
            
            if self.token_stats:
                f.write("TOKEN STATISTICS:\n")
                f.write("-"*80 + "\n")
                for key, value in self.token_stats.items():
                    f.write(f"  {key}: {value}\n")
                if self.token_stats.get('articles_with_tokens', 0) > 0:
                    avg_tokens = self.token_stats['total_tokens'] / self.token_stats['articles_with_tokens']
                    f.write(f"  Average tokens per article: {avg_tokens:.1f}\n")
                f.write("\n")
            
            # Learning progress statistics
            if self.metrics['reward_progression']:
                f.write("LEARNING PROGRESS ANALYSIS:\n")
                f.write("-"*80 + "\n")
                
                rewards = self.metrics['reward_progression']
                f.write(f"  Total training steps: {len(rewards)}\n")
                f.write(f"  Initial reward (first 10 avg): {np.mean(rewards[:10]):.4f}\n")
                f.write(f"  Final reward (last 10 avg): {np.mean(rewards[-10:]):.4f}\n")
                f.write(f"  Overall improvement: {np.mean(rewards[-10:]) - np.mean(rewards[:10]):.4f}\n")
                
                if len(rewards) > 100:
                    # Calculate improvement rate
                    mid_point = len(rewards) // 2
                    first_half_avg = np.mean(rewards[:mid_point])
                    second_half_avg = np.mean(rewards[mid_point:])
                    f.write(f"  First half average: {first_half_avg:.4f}\n")
                    f.write(f"  Second half average: {second_half_avg:.4f}\n")
                    f.write(f"  Improvement rate: {((second_half_avg - first_half_avg) / first_half_avg * 100):.2f}%\n")
                f.write("\n")
            
            if self.metrics['comprehension_scores']:
                f.write("COMPREHENSION ANALYSIS:\n")
                f.write("-"*80 + "\n")
                
                comp_scores = self.metrics['comprehension_scores']
                f.write(f"  Average comprehension: {np.mean(comp_scores):.4f}\n")
                f.write(f"  Comprehension std dev: {np.std(comp_scores):.4f}\n")
                f.write(f"  Min comprehension: {np.min(comp_scores):.4f}\n")
                f.write(f"  Max comprehension: {np.max(comp_scores):.4f}\n")
                f.write(f"  Median comprehension: {np.median(comp_scores):.4f}\n")
                
                if len(comp_scores) > 100:
                    initial_comp = np.mean(comp_scores[:10])
                    final_comp = np.mean(comp_scores[-10:])
                    f.write(f"  Initial comprehension: {initial_comp:.4f}\n")
                    f.write(f"  Final comprehension: {final_comp:.4f}\n")
                    f.write(f"  Comprehension improvement: {final_comp - initial_comp:.4f}\n")
                f.write("\n")
            
            f.write("TOP 10 BEST PERFORMING ARTICLES:\n")
            f.write("-"*80 + "\n")
            sorted_articles = sorted(self.article_logs, key=lambda x: x['reward'], reverse=True)[:10]
            for i, article in enumerate(sorted_articles, 1):
                f.write(f"  {i}. {article['title'][:60]}...\n")
                f.write(f"     Reward: {article['reward']:.4f}, Tokens: {article['tokens_processed']}\n")
                if 'comprehension' in article:
                    f.write(f"     Comprehension: {article['comprehension']:.4f}\n")
                f.write("\n")
            
            # Sample interactions
            if len(self.article_logs) > 0:
                f.write("SAMPLE INTERACTIONS:\n")
                f.write("="*80 + "\n\n")
                
                # Show first interaction
                f.write("FIRST INTERACTION (Article 0):\n")
                f.write("-"*80 + "\n")
                first_log = self.article_logs[0]
                f.write(f"Title: {first_log['title']}\n")
                f.write(f"Input Text (first 500 chars):\n{first_log.get('input_text', '')[:500]}...\n\n")
                f.write(f"Agent Response:\n{first_log.get('agent_response', 'No response recorded')[:500]}...\n\n")
                f.write(f"Reward: {first_log['reward']:.4f}\n")
                if 'comprehension' in first_log:
                    f.write(f"Comprehension: {first_log['comprehension']:.4f}\n")
                f.write("\n")
                
                # Show middle interaction
                if len(self.article_logs) > 10:
                    mid_idx = len(self.article_logs) // 2
                    f.write(f"MIDDLE INTERACTION (Article {mid_idx}):\n")
                    f.write("-"*80 + "\n")
                    mid_log = self.article_logs[mid_idx]
                    f.write(f"Title: {mid_log['title']}\n")
                    f.write(f"Input Text (first 500 chars):\n{mid_log.get('input_text', '')[:500]}...\n\n")
                    f.write(f"Agent Response:\n{mid_log.get('agent_response', 'No response recorded')[:500]}...\n\n")
                    f.write(f"Reward: {mid_log['reward']:.4f}\n")
                    if 'comprehension' in mid_log:
                        f.write(f"Comprehension: {mid_log['comprehension']:.4f}\n")
                    f.write("\n")
                
                # Show last interaction
                f.write(f"FINAL INTERACTION (Article {len(self.article_logs)-1}):\n")
                f.write("-"*80 + "\n")
                last_log = self.article_logs[-1]
                f.write(f"Title: {last_log['title']}\n")
                f.write(f"Input Text (first 500 chars):\n{last_log.get('input_text', '')[:500]}...\n\n")
                f.write(f"Agent Response:\n{last_log.get('agent_response', 'No response recorded')[:500]}...\n\n")
                f.write(f"Reward: {last_log['reward']:.4f}\n")
                if 'comprehension' in last_log:
                    f.write(f"Comprehension: {last_log['comprehension']:.4f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
        
        logging.info(f"Training summary report saved to {report_path}")
        
        # Also print to console
        print("\n" + "="*80)
        print("TRAINING COMPLETE - Summary saved to:", report_path)
        print("="*80)


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
        Train agent using next-token prediction on article content.

        This implements proper next-token prediction as used in LLM training:
        - For each position i in the token sequence, the model sees tokens[0:i]
        - And is trained to predict tokens[i] (the next token)
        - This creates multiple training examples from a single text

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

            if not self.tokenizer:
                print("[Warning] No tokenizer available, skipping training")
                return {'reward': 0.0, 'error': 'No tokenizer', 'tokens': []}

            try:
                # Encode text to token IDs (limit to first 512 tokens for efficiency)
                tokens = self.tokenizer.encode(content[:4000], add_bos=True, add_eos=True)

                if len(tokens) < 2:
                    print("[Warning] Text too short for next-token prediction")
                    return {'reward': 0.0, 'tokens': []}

                print(f"[Next-Token Training] Processing {len(tokens)} tokens")

            except Exception as e:
                print(f"[Warning] Tokenization failed: {e}")
                return {'reward': 0.0, 'error': str(e), 'tokens': []}

            # Next-token prediction training
            # For each position i, we predict token[i] given tokens[0:i]
            total_reward = 0.0
            correct_predictions = 0
            num_predictions = 0

            # Sample positions for training (to avoid overwhelming the system)
            # In a real LLM, we'd train on ALL positions, but for efficiency we sample
            max_training_examples = min(50, len(tokens) - 1)  # Train on up to 50 examples per article
            training_positions = np.random.choice(
                range(1, len(tokens)),
                size=min(max_training_examples, len(tokens) - 1),
                replace=False
            )
            training_positions = sorted(training_positions)

            for i, position in enumerate(training_positions):
                # Context: all tokens up to (but not including) position
                context_tokens = tokens[:position]
                # Target: the token at position
                target_token = tokens[position]

                # Convert context tokens to text for the agent
                context_text = self.tokenizer.decode(context_tokens)
                target_text = self.tokenizer.decode([target_token])

                # Send context to agent for next-token prediction
                command = f"process_text: {context_text}\n"
                content_bytes = command.encode('utf-8', errors='replace')
                self.process.stdin.write(content_bytes)
                self.process.stdin.flush()

                # Give the process a moment to respond
                time.sleep(0.05)  # Shorter delay for efficiency

                # Read agent's prediction
                stdout, stderr = self._read_streams()

                # Parse prediction and compare with target
                predicted_token = None
                if stdout and "NEXT_WORD_PREDICTION:" in stdout:
                    try:
                        # Extract the predicted text
                        pred_line = [line for line in stdout.split('\n') if 'NEXT_WORD_PREDICTION:' in line][0]
                        predicted_text = pred_line.split('NEXT_WORD_PREDICTION:')[1].strip()

                        # Calculate reward based on similarity to target
                        # Simple reward: 1.0 if exact match, partial credit for similarity
                        if predicted_text.lower() == target_text.lower():
                            reward = 1.0
                            correct_predictions += 1
                        elif predicted_text and target_text.lower() in predicted_text.lower():
                            reward = 0.5
                        elif predicted_text:
                            reward = 0.1
                        else:
                            reward = 0.0

                        # Send reward signal to agent for learning
                        reward_command = f"REWARD_SIGNAL: {reward}\n"
                        self.process.stdin.write(reward_command.encode('utf-8'))
                        self.process.stdin.flush()

                        total_reward += reward
                        num_predictions += 1

                        # Log progress every 10 predictions
                        if (i + 1) % 10 == 0:
                            accuracy = correct_predictions / num_predictions if num_predictions > 0 else 0
                            print(f"  [{i+1}/{len(training_positions)}] Accuracy: {accuracy:.2%}, "
                                  f"Avg reward: {total_reward/num_predictions:.3f}")

                    except Exception as e:
                        print(f"[Warning] Failed to parse prediction: {e}")
                        num_predictions += 1

                # Update stats
                self.stats['tokenizer']['tokens_processed'] = \
                    self.stats['tokenizer'].get('tokens_processed', 0) + len(context_tokens)

            # Calculate final metrics
            avg_reward = total_reward / max(1, num_predictions)
            accuracy = correct_predictions / max(1, num_predictions)

            print(f"[Next-Token Training Complete]")
            print(f"  Training examples: {num_predictions}")
            print(f"  Correct predictions: {correct_predictions}/{num_predictions} ({accuracy:.1%})")
            print(f"  Average reward: {avg_reward:.3f}")

            # Update stats
            self.stats['browsing']['urls_visited'] += 1
            self.stats['browsing']['pages_read'] += 1

            return {
                'reward': avg_reward,
                'tokens': tokens,
                'num_tokens': len(tokens),
                'num_training_examples': num_predictions,
                'accuracy': accuracy,
                'correct_predictions': correct_predictions
            }

        except (IOError, BrokenPipeError) as e:
            print(f"[Error] Communication error with NeuroGen process: {e}")
            self.shutdown()
            return {'reward': 0.0, 'error': str(e), 'tokens': []}
        except Exception as e:
            print(f"[Error] An unexpected error occurred in browse_page: {e}")
            import traceback
            traceback.print_exc()
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
        
        # Initialize logger
        self.logger = TrainingLogger()
    
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
            
            # Log epoch metrics
            self.logger.log_epoch_metrics(
                self.epoch, 
                len(epoch_articles), 
                avg_epoch_reward, 
                epoch_reward
            )
            
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
        
        # Save detailed logs
        self.logger.save_detailed_logs()
        
        # Generate visualizations and summary report
        training_config = {
            'num_epochs': num_epochs,
            'articles_per_epoch': articles_per_epoch,
            'total_articles': total_articles,
            'checkpoint_every': checkpoint_every,
            'total_time_minutes': elapsed / 60
        }
        
        final_stats = {
            'articles_processed': self.article_idx,
            'total_reward': self.total_reward,
            'avg_reward_per_article': self.total_reward / self.article_idx,
            'final_curiosity_score': self.agent.curiosity_score
        }
        
        self.logger.generate_visualizations()
        self.logger.generate_summary_report(training_config, final_stats)
        
        return self.metrics
    
    def _process_article(self, article):
        """
        Process a single Wikipedia article using next-token prediction.

        Args:
            article: Article dictionary

        Returns:
            Reward value
        """
        print(f"\n[Article] {article['title']} ({article['length']} chars)")

        # Train using next-token prediction
        result = self.agent.browse_page(
            url=article['url'],
            content=article['text']
        )

        # Display training metrics
        if result.get('accuracy') is not None:
            accuracy = result['accuracy']
            num_examples = result.get('num_training_examples', 0)
            correct = result.get('correct_predictions', 0)
            print(f"[Training Metrics] {correct}/{num_examples} correct ({accuracy:.1%} accuracy)")

        # Calculate reward based on multiple factors
        reward = result['reward']

        # Bonus for high accuracy in next-token prediction
        if result.get('accuracy') is not None:
            accuracy_bonus = result['accuracy'] * 0.2  # Up to 0.2 bonus for perfect accuracy
            reward += accuracy_bonus

        # Additional reward for longer, more informative articles
        length_bonus = min(0.1, article['length'] / 10000)
        reward += length_bonus

        # Reward for maintaining curiosity (exploration)
        curiosity_bonus = self.agent.curiosity_score * 0.1
        reward += curiosity_bonus

        reward = np.clip(reward, 0, 1.5)  # Allow higher rewards for good performance

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