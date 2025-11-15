// ============================================================================
// TEXT TOKENIZER HEADER
// File: include/NeuroGen/TextTokenizer.h
// ============================================================================

#ifndef TEXT_TOKENIZER_H
#define TEXT_TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

/**
 * @brief Text Tokenizer for Natural Language Processing
 * 
 * This class provides text tokenization capabilities for the autonomous learning agent.
 * It converts raw text into numerical tokens that can be processed by neural networks.
 */
class TextTokenizer {
public:
    /**
     * @brief Token structure
     */
    struct Token {
        int id;
        std::string text;
        int frequency;
        
        Token() : id(-1), frequency(0) {}
        Token(int token_id, const std::string& token_text, int freq = 1) 
            : id(token_id), text(token_text), frequency(freq) {}
    };

    /**
     * @brief Constructor with vocabulary size
     * @param vocab_size Maximum vocabulary size
     */
    explicit TextTokenizer(size_t vocab_size = 10000);
    
    /**
     * @brief Destructor
     */
    ~TextTokenizer() = default;
    
    /**
     * @brief Tokenize text into token IDs
     * @param text Input text to tokenize
     * @return Vector of token IDs
     */
    std::vector<int> tokenize(const std::string& text);
    
    /**
     * @brief Convert token IDs back to text
     * @param token_ids Vector of token IDs
     * @return Reconstructed text
     */
    std::string detokenize(const std::vector<int>& token_ids);
    
    /**
     * @brief Get token by ID
     * @param token_id Token ID to look up
     * @return Token object (empty if not found)
     */
    Token getToken(int token_id) const;
    
    /**
     * @brief Get token ID by text
     * @param text Token text to look up
     * @return Token ID (-1 if not found)
     */
    int getTokenId(const std::string& text) const;
    
    /**
     * @brief Add new token to vocabulary
     * @param text Token text
     * @return Token ID
     */
    int addToken(const std::string& text);
    
    /**
     * @brief Get vocabulary size
     * @return Current vocabulary size
     */
    size_t getVocabularySize() const;
    
    /**
     * @brief Get maximum vocabulary size
     * @return Maximum vocabulary size
     */
    size_t getMaxVocabularySize() const;
    
    /**
     * @brief Build vocabulary from text corpus
     * @param texts Vector of text strings
     */
    void buildVocabulary(const std::vector<std::string>& texts);
    
    /**
     * @brief Save vocabulary to file
     * @param filename Path to save vocabulary
     * @return Success status
     */
    bool saveVocabulary(const std::string& filename) const;
    
    /**
     * @brief Load vocabulary from file
     * @param filename Path to load vocabulary from
     * @return Success status
     */
    bool loadVocabulary(const std::string& filename);
    
    /**
     * @brief Clear vocabulary
     */
    void clearVocabulary();
    
    /**
     * @brief Get special token IDs
     */
    int getPadTokenId() const { return pad_token_id_; }
    int getUnkTokenId() const { return unk_token_id_; }
    int getBosTokenId() const { return bos_token_id_; }
    int getEosTokenId() const { return eos_token_id_; }

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    size_t max_vocab_size_;
    size_t current_vocab_size_;
    
    // Token mappings
    std::unordered_map<std::string, int> text_to_id_;
    std::unordered_map<int, Token> id_to_token_;
    
    // Special token IDs
    int pad_token_id_;  // Padding token
    int unk_token_id_;  // Unknown token
    int bos_token_id_;  // Beginning of sequence
    int eos_token_id_;  // End of sequence
    
    int next_token_id_;
    
    // ========================================================================
    // INTERNAL METHODS
    // ========================================================================
    
    /**
     * @brief Initialize special tokens
     */
    void initializeSpecialTokens();
    
    /**
     * @brief Split text into words
     * @param text Input text
     * @return Vector of words
     */
    std::vector<std::string> splitText(const std::string& text) const;
    
    /**
     * @brief Clean and preprocess text
     * @param text Input text
     * @return Cleaned text
     */
    std::string preprocessText(const std::string& text) const;
    
    /**
     * @brief Check if vocabulary is full
     * @return True if vocabulary is at maximum capacity
     */
    bool isVocabularyFull() const;
};

#endif // TEXT_TOKENIZER_H