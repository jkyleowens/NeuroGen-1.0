#!/usr/bin/env python3
"""
Interactive NLP Agent with Token Decoding

This script wraps the C++ NeuroGen agent and decodes generated tokens using SentencePiece.
Run this instead of the raw C++ executable to see decoded text responses.

Usage:
    python interactive_decode.py
"""

import sys
import subprocess
import threading
import queue
import re
from pathlib import Path
from sentencepiece_module import TokenizerModule


class InteractiveAgent:
    """Interactive agent with automatic token decoding"""

    def __init__(self, executable='./NeuroGen', tokenizer_model='./nlp_agent_tokenizer.model'):
        self.executable = executable
        self.tokenizer_model = tokenizer_model
        self.tokenizer = None
        self.process = None
        self.output_queue = queue.Queue()

    def load_tokenizer(self):
        """Load the SentencePiece tokenizer"""
        if not Path(self.tokenizer_model).exists():
            print(f"⚠️  Warning: Tokenizer model not found: {self.tokenizer_model}")
            print("   Token IDs will be shown but not decoded to text")
            return False

        try:
            self.tokenizer = TokenizerModule()
            self.tokenizer.load_model(self.tokenizer_model)
            print(f"✅ Loaded tokenizer: {self.tokenizer_model}")
            print(f"   Vocabulary size: {self.tokenizer.get_vocab_size()}")
            return True
        except Exception as e:
            print(f"⚠️  Error loading tokenizer: {e}")
            return False

    def _read_output(self, pipe, queue):
        """Read output from process and put in queue"""
        for line in iter(pipe.readline, b''):
            queue.put(line.decode('utf-8', errors='replace'))
        pipe.close()

    def start(self):
        """Start the C++ agent process"""
        if not Path(self.executable).exists():
            print(f"❌ Error: Executable not found: {self.executable}")
            print("   Please build the project first: make")
            return False

        try:
            self.process = subprocess.Popen(
                [self.executable],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1
            )

            # Start thread to read output
            self.output_thread = threading.Thread(
                target=self._read_output,
                args=(self.process.stdout, self.output_queue)
            )
            self.output_thread.daemon = True
            self.output_thread.start()

            return True

        except Exception as e:
            print(f"❌ Error starting agent: {e}")
            return False

    def process_output(self, timeout=0.1):
        """Process and display agent output, decoding tokens when found"""
        output_lines = []

        # Collect all available output
        import time
        end_time = time.time() + timeout
        while time.time() < end_time:
            try:
                line = self.output_queue.get(timeout=0.01)
                output_lines.append(line)
            except queue.Empty:
                continue

        # Process each line
        for line in output_lines:
            # Check for TOKEN_IDS output
            token_match = re.search(r'TOKEN_IDS:([0-9,]+)', line)
            if token_match and self.tokenizer:
                token_ids_str = token_match.group(1)
                try:
                    # Parse token IDs
                    token_ids = [int(tid) for tid in token_ids_str.split(',')]

                    # Decode using sentencepiece
                    decoded_text = self.tokenizer.decode(token_ids)

                    # Display the line as-is
                    print(line, end='')

                    # Add decoded text
                    print(f"📝 Decoded Text: {decoded_text}")
                    continue

                except Exception as e:
                    print(f"⚠️  Error decoding tokens: {e}")

            # Print line as-is if not a token line
            print(line, end='')

    def send_input(self, text):
        """Send input to the agent"""
        if not self.process or self.process.poll() is not None:
            print("❌ Agent process not running")
            return False

        try:
            self.process.stdin.write((text + '\n').encode('utf-8'))
            self.process.stdin.flush()
            return True
        except Exception as e:
            print(f"❌ Error sending input: {e}")
            return False

    def run_interactive(self):
        """Run interactive session"""
        print("\n" + "="*60)
        print("🧠 INTERACTIVE NLP AGENT WITH TOKEN DECODING")
        print("="*60)
        print("\nType your text to interact with the agent.")
        print("Type 'quit' or 'exit' to stop.\n")
        print("="*60)

        # Load tokenizer
        self.load_tokenizer()

        # Start agent
        if not self.start():
            return

        # Give agent time to initialize
        import time
        time.sleep(2)

        # Show initial output
        self.process_output(timeout=1.0)

        # Interactive loop
        try:
            while True:
                # Get user input
                try:
                    user_input = input("\n> ")
                except EOFError:
                    break

                if user_input.lower() in ['quit', 'exit']:
                    break

                if not user_input.strip():
                    continue

                # Send to agent
                if not self.send_input(user_input):
                    break

                # Wait for and display output
                time.sleep(0.5)  # Give agent time to process
                self.process_output(timeout=2.0)

        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")

        finally:
            self.shutdown()

    def shutdown(self):
        """Shutdown the agent"""
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                self.process.kill()
            print("\n✅ Agent shut down")


def main():
    """Main entry point"""
    agent = InteractiveAgent()
    agent.run_interactive()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
