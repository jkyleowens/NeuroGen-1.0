// ============================================================================
// TEXT TOKENIZER IMPLEMENTATION
// File: src/TextTokenizer.cpp
// ============================================================================

#include "NeuroGen/TextTokenizer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <regex>

// ============================================================================
// CONSTRUCTION AND INITIALIZATION
// ============================================================================

TextTokenizer::TextTokenizer(size_t vocab_size) 
    : max_vocab_size_(vocab_size),
      current_vocab_size_(0),
      next_token_id_(0) {
    
    initializeSpecialTokens();
    
    std::cout << "📝 TextTokenizer: Initialized with max vocabulary size: " 
              << max_vocab_size_ << std::endl;
}

void TextTokenizer::initializeSpecialTokens() {
    // Add special tokens
    pad_token_id_ = addToken("<PAD>");
    unk_token_id_ = addToken("<UNK>");
    bos_token_id_ = addToken("<BOS>");
    eos_token_id_ = addToken("<EOS>");
    
    std::cout << "📝 TextTokenizer: Initialized special tokens" << std::endl;
}

// ============================================================================
// CORE TOKENIZATION METHODS
// ============================================================================

std::vector<int> TextTokenizer::tokenize(const std::string& text) {
    std::vector<int> token_ids;
    
    if (text.empty()) {
        return token_ids;
    }
    
    // Preprocess and split text
    std::string cleaned_text = preprocessText(text);
    std::vector<std::string> words = splitText(cleaned_text);
    
    // Add beginning of sequence token
    token_ids.push_back(bos_token_id_);
    
    // Convert words to token IDs
    for (const std::string& word : words) {
        int token_id = getTokenId(word);
        if (token_id == -1) {
            // Unknown word - use UNK token or add to vocabulary
            if (!isVocabularyFull()) {
                token_id = addToken(word);
            } else {
                token_id = unk_token_id_;
            }
        }
        token_ids.push_back(token_id);
    }
    
    // Add end of sequence token
    token_ids.push_back(eos_token_id_);
    
    return token_ids;
}

std::string TextTokenizer::detokenize(const std::vector<int>& token_ids) {
    std::ostringstream result;
    
    for (size_t i = 0; i < token_ids.size(); ++i) {
        int token_id = token_ids[i];
        
        // Skip special tokens
        if (token_id == pad_token_id_ || token_id == bos_token_id_ || token_id == eos_token_id_) {
            continue;
        }
        
        Token token = getToken(token_id);
        if (token.id != -1) {
            if (i > 0 && token_ids[i-1] != bos_token_id_) {
                result << " ";
            }
            result << token.text;
        }
    }
    
    return result.str();
}

// ============================================================================
// VOCABULARY MANAGEMENT
// ============================================================================

TextTokenizer::Token TextTokenizer::getToken(int token_id) const {
    auto it = id_to_token_.find(token_id);
    if (it != id_to_token_.end()) {
        return it->second;
    }
    return Token(); // Return empty token if not found
}

int TextTokenizer::getTokenId(const std::string& text) const {
    auto it = text_to_id_.find(text);
    if (it != text_to_id_.end()) {
        return it->second;
    }
    return -1; // Not found
}

int TextTokenizer::addToken(const std::string& text) {
    // Check if token already exists
    int existing_id = getTokenId(text);
    if (existing_id != -1) {
        // Update frequency
        id_to_token_[existing_id].frequency++;
        return existing_id;
    }
    
    // Check vocabulary capacity
    if (isVocabularyFull()) {
        std::cout << "⚠️ TextTokenizer: Vocabulary full, cannot add token: " << text << std::endl;
        return unk_token_id_;
    }
    
    // Add new token
    int new_id = next_token_id_++;
    Token new_token(new_id, text, 1);
    
    text_to_id_[text] = new_id;
    id_to_token_[new_id] = new_token;
    current_vocab_size_++;
    
    return new_id;
}

void TextTokenizer::buildVocabulary(const std::vector<std::string>& texts) {
    std::cout << "📝 TextTokenizer: Building vocabulary from " << texts.size() << " texts..." << std::endl;
    
    // Count word frequencies
    std::unordered_map<std::string, int> word_counts;
    
    for (const std::string& text : texts) {
        std::string cleaned_text = preprocessText(text);
        std::vector<std::string> words = splitText(cleaned_text);
        
        for (const std::string& word : words) {
            word_counts[word]++;
        }
    }
    
    // Sort words by frequency
    std::vector<std::pair<std::string, int>> sorted_words;
    for (const auto& [word, count] : word_counts) {
        sorted_words.emplace_back(word, count);
    }
    
    std::sort(sorted_words.begin(), sorted_words.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second; // Sort by frequency (descending)
              });
    
    // Add most frequent words to vocabulary
    for (const auto& [word, count] : sorted_words) {
        if (isVocabularyFull()) {
            break;
        }
        addToken(word);
    }
    
    std::cout << "📝 TextTokenizer: Built vocabulary with " << current_vocab_size_ 
              << " tokens from " << word_counts.size() << " unique words" << std::endl;
}

// ============================================================================
// PERSISTENCE
// ============================================================================

bool TextTokenizer::saveVocabulary(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ TextTokenizer: Failed to open file for writing: " << filename << std::endl;
        return false;
    }
    
    // Write header
    file.write("VOCAB", 5);
    uint32_t version = 1;
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write vocabulary size
    uint32_t vocab_size = static_cast<uint32_t>(current_vocab_size_);
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    
    // Write special tokens
    file.write(reinterpret_cast<const char*>(&pad_token_id_), sizeof(pad_token_id_));
    file.write(reinterpret_cast<const char*>(&unk_token_id_), sizeof(unk_token_id_));
    file.write(reinterpret_cast<const char*>(&bos_token_id_), sizeof(bos_token_id_));
    file.write(reinterpret_cast<const char*>(&eos_token_id_), sizeof(eos_token_id_));
    
    // Write tokens
    for (const auto& [id, token] : id_to_token_) {
        uint32_t token_id = static_cast<uint32_t>(token.id);
        uint32_t text_len = static_cast<uint32_t>(token.text.length());
        uint32_t frequency = static_cast<uint32_t>(token.frequency);
        
        file.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
        file.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
        file.write(token.text.c_str(), text_len);
        file.write(reinterpret_cast<const char*>(&frequency), sizeof(frequency));
    }
    
    file.close();
    std::cout << "💾 TextTokenizer: Saved vocabulary to " << filename << std::endl;
    return true;
}

bool TextTokenizer::loadVocabulary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "❌ TextTokenizer: Failed to open file for reading: " << filename << std::endl;
        return false;
    }
    
    // Read header
    char header[6];
    file.read(header, 5);
    header[5] = '\0';
    if (std::string(header) != "VOCAB") {
        std::cerr << "❌ TextTokenizer: Invalid vocabulary file format" << std::endl;
        return false;
    }
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != 1) {
        std::cerr << "❌ TextTokenizer: Unsupported vocabulary file version: " << version << std::endl;
        return false;
    }
    
    // Clear existing vocabulary
    clearVocabulary();
    
    // Read vocabulary size
    uint32_t vocab_size;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
    
    // Read special tokens
    file.read(reinterpret_cast<char*>(&pad_token_id_), sizeof(pad_token_id_));
    file.read(reinterpret_cast<char*>(&unk_token_id_), sizeof(unk_token_id_));
    file.read(reinterpret_cast<char*>(&bos_token_id_), sizeof(bos_token_id_));
    file.read(reinterpret_cast<char*>(&eos_token_id_), sizeof(eos_token_id_));
    
    // Read tokens
    for (uint32_t i = 0; i < vocab_size; ++i) {
        uint32_t token_id, text_len, frequency;
        
        file.read(reinterpret_cast<char*>(&token_id), sizeof(token_id));
        file.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));
        
        std::string text(text_len, '\0');
        file.read(&text[0], text_len);
        
        file.read(reinterpret_cast<char*>(&frequency), sizeof(frequency));
        
        Token token(static_cast<int>(token_id), text, static_cast<int>(frequency));
        text_to_id_[text] = static_cast<int>(token_id);
        id_to_token_[static_cast<int>(token_id)] = token;
    }
    
    current_vocab_size_ = vocab_size;
    next_token_id_ = static_cast<int>(vocab_size);
    
    file.close();
    std::cout << "📂 TextTokenizer: Loaded vocabulary from " << filename 
              << " (" << vocab_size << " tokens)" << std::endl;
    return true;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

size_t TextTokenizer::getVocabularySize() const {
    return current_vocab_size_;
}

size_t TextTokenizer::getMaxVocabularySize() const {
    return max_vocab_size_;
}

void TextTokenizer::clearVocabulary() {
    text_to_id_.clear();
    id_to_token_.clear();
    current_vocab_size_ = 0;
    next_token_id_ = 0;
    
    // Re-initialize special tokens
    initializeSpecialTokens();
    
    std::cout << "🗑️ TextTokenizer: Cleared vocabulary" << std::endl;
}

bool TextTokenizer::isVocabularyFull() const {
    return current_vocab_size_ >= max_vocab_size_;
}

// ============================================================================
// TEXT PROCESSING HELPERS
// ============================================================================

std::string TextTokenizer::preprocessText(const std::string& text) const {
    std::string result = text;
    
    // Convert to lowercase
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    
    // Remove extra whitespace
    std::regex whitespace_regex(R"(\s+)");
    result = std::regex_replace(result, whitespace_regex, " ");
    
    // Trim leading and trailing whitespace
    result.erase(0, result.find_first_not_of(" \t\n\r"));
    result.erase(result.find_last_not_of(" \t\n\r") + 1);
    
    return result;
}

std::vector<std::string> TextTokenizer::splitText(const std::string& text) const {
    std::vector<std::string> words;
    std::istringstream stream(text);
    std::string word;
    
    while (stream >> word) {
        // Simple word-level tokenization
        // Remove punctuation from the end
        while (!word.empty() && std::ispunct(word.back())) {
            word.pop_back();
        }
        
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    
    return words;
}