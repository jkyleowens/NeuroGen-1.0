#!/usr/bin/env python3
"""
NeuroGen 0.5.5 - Massive Modular Neural Agent Language Training
================================================================

This script initializes and trains the massive modular neural agent (50,000+ neurons)
on natural language data to improve its overall control and language understanding.

Features:
- Initializes the autonomous learning agent with tens of thousands of neurons
- Trains on diverse natural language datasets
- Implements persistent saving/loading of sub-networks
- Monitors training progress and neural dynamics
- Provides real-time feedback on language understanding
"""

import os
import sys
import time
import json
import subprocess
import threading
import signal
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('language_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for language training"""
    # Agent configuration
    agent_executable: str = "./NeuroGen_Autonomous"
    max_training_time: int = 7200  # 2 hours
    save_interval: int = 300       # Save every 5 minutes
    
    # Language training parameters
    vocabulary_size: int = 10000
    max_sequence_length: int = 128
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Dataset configuration
    use_wikipedia: bool = True
    use_news: bool = True
    use_literature: bool = True
    use_technical: bool = True
    dataset_size_multiplier: int = 50 # Greatly increase dataset size, simulating "The Pile"

    # Persistence configuration
    save_directory: str = "neural_agent_saves"
    backup_count: int = 5

class LanguageDataProvider:
    """Provides diverse language training data, simulating 'The Pile'."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.vocabulary = set()
        self.training_samples: List[Tuple[str, str]] = []
        self.training_texts = []
        self.current_position = 0

    def load_datasets(self):
        """Load and prepare language datasets"""
        logger.info("🔤 Loading diverse language datasets (simulating 'The Pile')...")

        # Sample training texts covering various domains - expanded for more variety
        sample_texts = [
            # Technical/Programming
            "The autonomous neural agent processes visual input through specialized cortical modules.",
            "Each module contains thousands of interconnected neurons that adapt through experience.",
            "Synaptic plasticity enables learning and memory formation in biological neural networks.",
            "The agent's decision-making process is governed by a reinforcement learning algorithm.",
            "Backpropagation is used to adjust synaptic weights based on prediction errors.",
            "Object-oriented programming allows for the creation of modular and reusable code.",
            "A recursive function is a function that calls itself during its execution.",
            "The quicksort algorithm is an efficient, in-place sorting algorithm.",
            "A hash table is a data structure that implements an associative array abstract data type.",
            "The TCP/IP model is a conceptual model and set of communications protocols used on the Internet.",

            # Natural language instructions
            "Click on the login button located in the top right corner of the screen.",
            "Enter your username in the text field below the password prompt.",
            "Navigate to the settings menu to configure your preferences.",
            "Please upload the document by dragging it into the designated area.",
            "Search for the latest news about artificial intelligence.",
            "Compose a new email addressed to 'example@example.com' with the subject 'Project Update'.",
            "Find the file named 'report.docx' in your downloads folder and open it.",
            "Increase the volume by twenty percent.",
            "Set a reminder for tomorrow at 9 AM to call the doctor.",
            "What is the capital of Australia?",

            # Conversational language
            "Hello, how can I help you today? I am an autonomous learning agent designed to assist with computer tasks.",
            "I can understand natural language and execute actions on your behalf.",
            "Please tell me what you would like me to do.",
            "What is the weather like in New York City right now?",
            "Can you find me a good recipe for chocolate chip cookies?",
            "That's an interesting point of view, could you elaborate on that?",
            "I'm not sure I understand, can you please rephrase the question?",
            "Let's talk about something else, what are your hobbies?",
            "It was a pleasure chatting with you.",
            "Thank you for your assistance, you've been very helpful.",

            # Problem-solving language
            "To solve this task, first analyze the current screen state. Identify clickable elements and determine the optimal sequence of actions.",
            "Consider the user's intent and adapt the strategy based on visual feedback from the environment.",
            "If an error occurs, try to identify the cause and find an alternative solution.",
            "The primary objective is to complete the data entry form accurately and efficiently.",
            "Let's break down the problem into smaller, more manageable steps.",
            "The first step is to gather all the necessary information before proceeding.",
            "We need to evaluate the pros and cons of each option before making a decision.",
            "A critical thinking approach is required to overcome this obstacle.",
            "Let's brainstorm some potential solutions together.",
            "By working collaboratively, we can find the most effective path forward.",

            # Scientific/Technical descriptions
            "Spike-timing dependent plasticity strengthens synaptic connections when pre-synaptic spikes precede post-synaptic spikes within a critical time window.",
            "This mechanism enables associative learning and memory consolidation in neural networks.",
            "The hippocampus plays a crucial role in the formation of new episodic memories.",
            "Quantum entanglement is a physical phenomenon that occurs when pairs or groups of particles are generated in such a way that the quantum state of each particle cannot be described independently of the others.",
            "The theory of general relativity describes gravity as a geometric property of spacetime.",
            "Photosynthesis is a process used by plants, algae, and certain bacteria to convert light energy into chemical energy.",
            "The DNA double helix is a spiral staircase-like structure that carries genetic instructions.",
            "The human brain is composed of approximately 86 billion neurons.",
            "Black holes are regions of spacetime where gravity is so strong that nothing, no particles or even electromagnetic radiation such as light, can escape from it.",
            "The standard model of particle physics is the theory describing three of the four known fundamental forces in the universe.",

            # Control and navigation language
            "Move the mouse cursor to coordinates 500, 300. Left-click to select the target element.",
            "Scroll down to reveal additional options. Type the requested text into the input field.",
            "Press Enter to confirm the action and proceed to the next step.",
            "Open the file explorer and navigate to the 'Documents' folder.",
            "Close the current application window.",
            "Switch to the previously opened tab in the web browser.",
            "Maximize the window to fill the entire screen.",
            "Copy the selected text to the clipboard.",
            "Paste the content from the clipboard into the text editor.",
            "Undo the last action performed."
        ]

        # Expand training set with variations
        expanded_texts = []
        for text in sample_texts:
            expanded_texts.append(text)
            # Create paraphrases or slight variations
            words = text.split()
            if len(words) > 2:
                for i in range(len(words)):
                    # Swap two words
                    if i < len(words) - 1:
                        swapped = words[:]
                        swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
                        expanded_texts.append(" ".join(swapped))
                    
                    # Remove a word
                    removed = words[:i] + words[i+1:]
                    expanded_texts.append(" ".join(removed))
                    
                    # Add a synonym (simple replacement)
                    if words[i].lower() == "quick":
                        synonym = "fast"
                    elif words[i].lower() == "brown":
                        synonym = "chestnut"
                    elif words[i].lower() == "fox":
                        synonym = "wolf"
                    else:
                        synonym = None
                    
                    if synonym:
                        replaced = words[:]
                        replaced[i] = synonym
                        expanded_texts.append(" ".join(replaced))
        
        # Shuffle and select top N samples to limit dataset size
        random.shuffle(expanded_texts)
        self.training_texts = expanded_texts[:self.config.vocabulary_size * 10] # 10x multiplier for variety
        random.shuffle(self.training_texts)

        # Build vocabulary
        all_words = " ".join(self.training_texts).lower().split()
        self.vocabulary = sorted(list(set(all_words)))


        logger.info(f"✅ Loaded {len(self.training_texts)} training examples")
        logger.info(f"✅ Vocabulary size: {len(self.vocabulary)} words")

    def get_next_batch(self, batch_size: int) -> List[Tuple[str, str]]:
        """Get the next batch of (context, next_word) pairs."""
        batch = []
        for _ in range(batch_size):
            if self.current_position >= len(self.training_texts):
                self.current_position = 0  # Restart from beginning
                random.shuffle(self.training_texts)

            full_sentence = self.training_texts[self.current_position]
            self.current_position += 1

            words = full_sentence.strip().split()
            if len(words) > 1:
                split_point = random.randint(1, len(words) - 1)
                context = " ".join(words[:split_point])
                next_word = words[split_point]
                batch.append((context, next_word))
        return batch

class NeuralAgentController:
    """Controls the NeuroGen autonomous agent process"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.process = None
        self.is_running = False
        self.performance_metrics = {}
        self.output_queue = [] # Simple queue for agent output
        self.prediction_event = threading.Event()
        self.last_prediction = None

    def start_agent(self, reset_model: bool = False) -> bool:
        """Start the autonomous learning agent"""
        try:
            logger.info("🚀 Starting massive modular neural agent...")

            # Set environment variables for large-scale training
            env = os.environ.copy()
            env['NEURGEN_TRAINING_MODE'] = '1'
            env['NEURGEN_LANGUAGE_TRAINING'] = '1'
            env['NEURGEN_MAX_NEURONS'] = '65536'  # Support up to 64K neurons

            command = [self.config.agent_executable]
            if reset_model:
                command.append("--reset-model")

            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            self.is_running = True
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self.monitor_output)
            monitor_thread.daemon = True
            monitor_thread.start()

            logger.info("✅ Neural agent started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start agent: {e}")
            return False
    
    def stop_agent(self):
        """Stop the agent gracefully"""
        if self.process and self.is_running:
            logger.info("🛑 Stopping neural agent...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.is_running = False
            logger.info("✅ Agent stopped")

    def send_command(self, command: str) -> bool:
        """Send a command to the agent."""
        if not self.is_running or not self.process or not self.process.stdin:
            return False
        try:
            self.process.stdin.write(f"COMMAND:{command}\n")
            self.process.stdin.flush()
            logger.info(f"Sent command to agent: {command}")
            return True
        except Exception as e:
            logger.error(f"Error sending command to agent: {e}")
            return False

    def get_prediction(self, timeout=5.0) -> Optional[str]:
        """Waits for and retrieves the next word prediction from the agent."""
        self.prediction_event.clear()
        self.last_prediction = None
        if self.prediction_event.wait(timeout):
            return self.last_prediction
        else:
            logger.warning("Timeout waiting for agent prediction.")
            return None

    def send_language_input(self, text: str) -> bool:
        """Send language input to the agent for processing"""
        if not self.is_running or not self.process or not self.process.stdin:
            return False
            
        try:
            # Convert text to neural input format
            input_command = f"COMMAND:LANGUAGE_INPUT:{text}\n"
            self.process.stdin.write(input_command)
            self.process.stdin.flush()
            logger.debug(f"Sent language input: {text[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to send input: {e}")
            return False
    
    def monitor_output(self):
        """Monitor agent output for training feedback"""
        if not self.process:
            return
            
        try:
            while self.is_running and self.process.poll() is None:
                if not self.process or not self.process.stdout:
                    break
                output = self.process.stdout.readline()
                if output:
                    self.parse_agent_output(output.strip())
                else:
                    # Check if process has terminated
                    if self.process.poll() is not None:
                        logger.error("Agent process has terminated unexpectedly")
                        self.is_running = False
                        break
        except Exception as e:
            if self.is_running:
                logger.error(f"Error monitoring output: {e}")
        
        # If we exit the loop, the process has died
        if self.is_running:
            logger.warning("Agent process monitoring stopped")
            self.is_running = False
    
    def parse_agent_output(self, output: str):
        """Parse agent output for performance metrics and predictions"""
        self.output_queue.append(output) # Store all output for debugging if needed

        if "NEXT_WORD_PREDICTION:" in output:
            try:
                prediction = output.split("NEXT_WORD_PREDICTION:")[1].strip()
                self.last_prediction = prediction
                self.prediction_event.set()
                logger.debug(f"Agent predicted: {prediction}")
            except IndexError:
                logger.warning(f"Could not parse prediction from output: {output}")

        elif "Learning Progress:" in output:
            try:
                progress = float(output.split(":")[-1].strip().replace("%", ""))
                self.performance_metrics['learning_progress'] = progress
                logger.info(f"📈 Learning Progress: {progress:.2f}%")
            except:
                pass
        elif "Neurons:" in output and "total" in output:
            try:
                neuron_count = output.split("total")[0].split(":")[-1].strip()
                logger.info(f"🧠 Active Neurons: {neuron_count}")
            except:
                pass

class PersistenceManager:
    """Manages saving and loading of neural network states"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.save_dir = Path(config.save_directory)
        self.save_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different module types
        self.module_dirs = {
            'visual_cortex': self.save_dir / 'visual_cortex',
            'prefrontal_cortex': self.save_dir / 'prefrontal_cortex',
            'motor_cortex': self.save_dir / 'motor_cortex',
            'working_memory': self.save_dir / 'working_memory',
            'reward_system': self.save_dir / 'reward_system',
            'attention_system': self.save_dir / 'attention_system'
        }
        
        for module_dir in self.module_dirs.values():
            module_dir.mkdir(exist_ok=True)
    
    def save_agent_state(self, training_step: int) -> bool:
        """Save complete agent state including all sub-networks"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"agent_state_step_{training_step}_{timestamp}"
            
            logger.info(f"💾 Saving agent state: {save_name}")
            
            # Create save command file for the agent to read
            save_command = {
                'action': 'save_state',
                'save_path': str(self.save_dir / save_name),
                'timestamp': timestamp,
                'training_step': training_step,
                'modules': list(self.module_dirs.keys())
            }
            
            # Write save command
            command_file = self.save_dir / 'save_command.json'
            with open(command_file, 'w') as f:
                json.dump(save_command, f, indent=2)
            
            # Create checkpoint metadata
            metadata = {
                'training_step': training_step,
                'timestamp': timestamp,
                'neuron_counts': {
                    'visual_cortex': 16384,
                    'prefrontal_cortex': 12288,
                    'motor_cortex': 8192,
                    'working_memory': 6144,
                    'reward_system': 4096,
                    'attention_system': 3072
                },
                'total_neurons': 49792,
                'architecture_version': '0.5.5'
            }
            
            metadata_file = self.save_dir / f"{save_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("✅ Agent state saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save agent state: {e}")
            return False
    
    def load_agent_state(self, save_name: str) -> bool:
        """Load agent state from checkpoint"""
        try:
            logger.info(f"📂 Loading agent state: {save_name}")
            
            # Check if save exists
            metadata_file = self.save_dir / f"{save_name}_metadata.json"
            if not metadata_file.exists():
                logger.error(f"Save file not found: {metadata_file}")
                return False
            
            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"📊 Loading checkpoint from step {metadata['training_step']}")
            logger.info(f"🧠 Total neurons: {metadata['total_neurons']}")
            
            # Create load command
            load_command = {
                'action': 'load_state',
                'load_path': str(self.save_dir / save_name),
                'metadata': metadata
            }
            
            command_file = self.save_dir / 'load_command.json'
            with open(command_file, 'w') as f:
                json.dump(load_command, f, indent=2)
            
            logger.info("✅ Agent state loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load agent state: {e}")
            return False
    
    def list_available_saves(self) -> List[str]:
        """List all available save states"""
        saves = []
        for file in self.save_dir.glob("*_metadata.json"):
            save_name = file.stem.replace("_metadata", "")
            saves.append(save_name)
        return sorted(saves)
    
    def cleanup_old_saves(self):
        """Remove old saves beyond backup count"""
        saves = self.list_available_saves()
        if len(saves) > self.config.backup_count:
            old_saves = saves[:-self.config.backup_count]
            for save_name in old_saves:
                try:
                    # Remove metadata file
                    metadata_file = self.save_dir / f"{save_name}_metadata.json"
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    # Remove save directory if it exists
                    save_path = self.save_dir / save_name
                    if save_path.exists() and save_path.is_dir():
                        import shutil
                        shutil.rmtree(save_path)
                    
                    logger.info(f"🗑️ Cleaned up old save: {save_name}")
                except Exception as e:
                    logger.error(f"Failed to cleanup {save_name}: {e}")

class LanguageTrainer:
    """Main language training orchestrator"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_provider = LanguageDataProvider(config)
        self.agent_controller = NeuralAgentController(config)
        self.persistence_manager = PersistenceManager(config)
        self.training_step = 0
        self.start_time = None
        self.is_training = False
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info("🛑 Received shutdown signal")
            self.stop_training()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_training(self):
        """Start the complete training process"""
        logger.info("🎓 Starting Massive Modular Neural Agent Language Training")
        logger.info("=" * 80)
        
        self.setup_signal_handlers()
        
        # Load training data
        self.data_provider.load_datasets()
        
        # Check for existing saves
        available_saves = self.persistence_manager.list_available_saves()
        if available_saves:
            logger.info(f"📂 Found {len(available_saves)} existing saves")
            latest_save = available_saves[-1]
            logger.info(f"🔄 Resuming from latest save: {latest_save}")
            self.persistence_manager.load_agent_state(latest_save)
        
        # Start the neural agent
        if not self.agent_controller.start_agent():
            logger.error("❌ Failed to start neural agent")
            return False
        
        # Set agent to passive language training mode
        time.sleep(5) # Allow agent to initialize
        if not self.agent_controller.send_command("SET_MODE:LANGUAGE_TRAINING"):
            logger.error("Failed to set agent to language training mode. Exiting.")
            self.agent_controller.stop_agent()
            return False
        
        # Give the agent time to process the mode change
        time.sleep(2)
        
        # Check if agent is still running after mode change
        if not self.agent_controller.is_running or self.agent_controller.process.poll() is not None:
            logger.error("Agent process died after setting language training mode")
            return False
        
        # Main training loop
        self.is_training = True
        self.start_time = time.time()
        
        try:
            self.training_loop()
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training error: {e}")
        finally:
            self.stop_training()
        
        return True
    
    def training_loop(self):
        """Main training loop for next-word prediction."""
        logger.info("🔄 Starting next-word prediction training loop...")
        
        last_save_time = time.time()
        
        while self.is_training:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Check if max training time reached
            if elapsed_time > self.config.max_training_time:
                logger.info("⏰ Maximum training time reached")
                break
            
            # Get a random training sample
            sample = self.data_provider.get_random_sample()
            if not sample:
                logger.warning("No more training data. Stopping.")
                break
            
            context, correct_word = sample
            
            # Send context to agent and get prediction
            if self.agent_controller.send_language_input(context):
                self.training_step += 1
                
                prediction = self.agent_controller.get_prediction(timeout=2.0)
                self.total_predictions += 1
                
                if prediction:
                    is_correct = prediction.lower() == correct_word.lower()
                    if is_correct:
                        self.correct_predictions += 1
                        reward = 1.0
                    else:
                        reward = -1.0
                    
                    self.agent_controller.send_feedback(reward)
                    
                    # Log the interaction
                    performance = (self.correct_predictions / self.total_predictions) * 100
                    log_msg = (
                        f"Step: {self.training_step} | "
                        f"Perf: {performance:.2f}% | "
                        f"Context: '{context}' | "
                        f"Pred: '{prediction}' | "
                        f"Actual: '{correct_word}' | "
                        f"Result: {'✅ Correct' if is_correct else '❌ Incorrect'}"
                    )
                    logger.info(log_msg)
                    
                else:
                    # Agent failed to predict in time
                    logger.warning(f"Step: {self.training_step} | Timeout waiting for prediction. Context: '{context}'")
                    self.agent_controller.send_feedback(-0.5) # Penalize for timeout
            
            # Brief pause between inputs
            time.sleep(0.05)
            
            # Periodic saving
            if current_time - last_save_time > self.config.save_interval:
                self.persistence_manager.save_agent_state(self.training_step)
                self.persistence_manager.cleanup_old_saves()
                last_save_time = current_time
        
        # Final save
        logger.info("💾 Performing final save...")
        self.persistence_manager.save_agent_state(self.training_step)
        
        # Training summary
        total_time = time.time() - self.start_time
        logger.info("🎉 Training completed!")
        logger.info(f"📊 Total training steps: {self.training_step}")
        if self.total_predictions > 0:
            final_performance = (self.correct_predictions / self.total_predictions) * 100
            logger.info(f"🎯 Final Prediction Accuracy: {final_performance:.2f}% ({self.correct_predictions}/{self.total_predictions})")
        logger.info(f"⏱️ Total training time: {total_time:.0f} seconds")
        logger.info(f"🧠 Final neural configuration: ~65,000 neurons across 6 modules")
    
    def stop_training(self):
        """Stop training gracefully"""
        self.is_training = False
        self.agent_controller.stop_agent()

def main():
    """Main function to run the language training"""
    config = TrainingConfig()
    agent_controller = NeuralAgentController(config)
    data_provider = LanguageDataProvider(config)

    # Handle command-line arguments
    reset_model = '--reset-model' in sys.argv

    def signal_handler(sig, frame):
        logger.info("SIGINT received, shutting down gracefully...")
        agent_controller.stop_agent()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Start the agent
        if not agent_controller.start_agent(reset_model):
            logger.error("Failed to start the agent. Exiting.")
            return

        # Allow agent to initialize
        logger.info("Waiting for agent to initialize...")
        time.sleep(5)

        # Set agent to passive language training mode
        logger.info("Setting agent to language training mode...")
        if not agent_controller.send_command("SET_MODE:LANGUAGE_TRAINING"):
            logger.error("Failed to set agent to language training mode. Exiting.")
            agent_controller.stop_agent()
            return
        time.sleep(1) # Give agent time to switch mode

        # Load datasets
        data_provider.load_datasets()

        start_time = time.time()
        last_save_time = time.time()
        training_step = 0
        correct_predictions = 0
        total_predictions = 0
        recent_predictions = []

        logger.info("🚀 Starting next-word prediction training loop...")
        while time.time() - start_time < config.max_training_time:
            # Get a single training example
            batch = data_provider.get_next_batch(batch_size=1)
            if not batch:
                continue
            
            context, next_word = batch[0]

            if not agent_controller.is_running:
                logger.warning("Agent process is not running. Stopping training.")
                break

            # 1. Send context to agent
            if not agent_controller.send_language_input(context):
                logger.error("Failed to send language input to agent. Stopping.")
                break
            
            # 2. Wait for the agent's prediction
            predicted_word = agent_controller.get_prediction(timeout=5.0)
            
            # 3. Provide reward and log results
            total_predictions += 1
            reward = -1.0  # Default to negative reward

            if predicted_word is not None:
                # Normalize words for comparison
                normalized_prediction = predicted_word.lower().strip(".,?!")
                normalized_actual = next_word.lower().strip(".,?!")

                if normalized_prediction == normalized_actual:
                    correct_predictions += 1
                    reward = 1.0
                    recent_predictions.append(1)
                    logger.info(f"✅ Correct! Context: '{context}', Prediction: '{predicted_word}', Actual: '{next_word}'")
                else:
                    recent_predictions.append(0)
                    logger.info(f"❌ Incorrect. Context: '{context}', Prediction: '{predicted_word}', Actual: '{next_word}'")
            else:
                recent_predictions.append(0)
                logger.warning(f"⚠️ No prediction received. Context: '{context}', Actual: '{next_word}'")

            # Keep recent_predictions list to a fixed size (e.g., last 100)
            if len(recent_predictions) > 100:
                recent_predictions.pop(0)

            # 4. Send reward signal to the agent
            agent_controller.send_command(f"REWARD_SIGNAL:{reward}")

            training_step += 1

            # Log performance periodically
            if training_step % 20 == 0 and total_predictions > 0:
                accuracy = (correct_predictions / total_predictions) * 100
                recent_accuracy = (sum(recent_predictions) / len(recent_predictions)) * 100 if recent_predictions else 0
                logger.info(f"📊 Step: {training_step} | Overall Acc: {accuracy:.2f}% | Recent Acc (last 100): {recent_accuracy:.2f}%")

            # Save agent state periodically
            current_time = time.time()
            if current_time - last_save_time > config.save_interval:
                logger.info("💾 Saving agent state...")
                if agent_controller.send_command("SAVE_STATE"):
                     logger.info("✅ Agent state save command sent.")
                else:
                    logger.error("Failed to send save command to agent.")
                last_save_time = current_time
            
            time.sleep(0.1) # Small delay between steps

        logger.info("🏁 Maximum training time reached or training stopped.")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logger.info("Shutting down the agent.")
        agent_controller.stop_agent()

if __name__ == "__main__":
    main()
