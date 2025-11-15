// ============================================================================
// LANGUAGE INTERFACE - NATURAL LANGUAGE PROCESSING INTERFACE
// File: include/NeuroGen/LanguageInterface.h
// ============================================================================

#ifndef LANGUAGE_INTERFACE_H
#define LANGUAGE_INTERFACE_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <fstream>
#include <sstream>

// Forward declarations for classes used by LanguageInterface
class TextTokenizer;
class LanguageAnalyzer;
class ResponseGenerator;

/**
 * @brief Language Interface for Natural Language Processing
 * 
 * This class provides a comprehensive interface for natural language input/output,
 * text processing, and language model interaction. It replaces the visual interface
 * with language-focused capabilities for reading, processing, and generating text.
 * 
 * Key Features:
 * - Real-time text input/output processing
 * - Multi-source text input (files, streams, interactive)
 * - Language importance mapping and attention guidance
 * - Text preprocessing and tokenization
 * - Response generation and output formatting
 * - Language metrics and performance tracking
 */
class LanguageInterface {
public:
    // ========================================================================
    // INPUT SOURCE TYPES
    // ========================================================================
    
    enum class InputSource {
        INTERACTIVE,        // Real-time user interaction
        FILE_READING,       // Reading from text files
        STREAM_PROCESSING,  // Processing text streams
        WEB_CONTENT,        // Web-based text content
        DOCUMENT_ANALYSIS,  // Document processing mode
        CONVERSATION,       // Multi-turn conversation
        AUTONOMOUS_READING  // Self-directed reading and learning
    };
    
    enum class ProcessingMode {
        REAL_TIME,          // Immediate processing
        BATCH_PROCESSING,   // Process text in batches
        STREAMING,          // Continuous stream processing
        INTERACTIVE_CHAT,   // Chat-like interaction
        DOCUMENT_MODE,      // Full document processing
        ANALYSIS_MODE       // Deep text analysis
    };

    // ========================================================================
    // LANGUAGE PROCESSING STRUCTURES
    // ========================================================================
    
    struct TextSegment {
        std::string text;
        std::string segment_type;  // "sentence", "paragraph", "section"
        float importance_score;
        std::vector<std::string> keywords;
        std::map<std::string, float> linguistic_features;
        std::chrono::system_clock::time_point timestamp;
        
        TextSegment() : importance_score(0.5f) {}
    };
    
    struct LanguageMetrics {
        float reading_speed_wpm;
        float comprehension_score;
        float response_quality;
        float vocabulary_coverage;
        float syntactic_complexity;
        float semantic_coherence;
        size_t words_processed;
        size_t sentences_processed;
        std::chrono::duration<float> processing_time;
        
        LanguageMetrics() : reading_speed_wpm(0), comprehension_score(0), 
                           response_quality(0), vocabulary_coverage(0),
                           syntactic_complexity(0), semantic_coherence(0),
                           words_processed(0), sentences_processed(0) {}
    };
    
    struct ConversationTurn {
        std::string speaker;        // "user" or "agent"
        std::string text;
        std::chrono::system_clock::time_point timestamp;
        float confidence_score;
        std::map<std::string, float> intent_scores;
        
        ConversationTurn() : confidence_score(0.0f) {}
    };

private:
    // ========================================================================
    // INTERNAL STATE
    // ========================================================================
    
    // Configuration
    InputSource current_input_source_;
    ProcessingMode current_processing_mode_;
    std::map<std::string, float> processing_parameters_;
    
    // Text processing state
    std::queue<TextSegment> input_queue_;
    std::queue<std::string> output_queue_;
    std::vector<ConversationTurn> conversation_history_;
    std::string current_context_;
    std::string pending_response_;
    
    // Language processing components
    // std::unique_ptr<TextTokenizer> tokenizer_;
    // std::unique_ptr<LanguageAnalyzer> analyzer_;
    // std::unique_ptr<ResponseGenerator> response_generator_;
    
    // Attention and importance mapping
    std::vector<float> text_importance_map_;
    std::map<std::string, float> keyword_importance_;
    std::vector<float> attention_guidance_;
    
    // Performance tracking
    LanguageMetrics current_metrics_;
    std::map<std::string, float> performance_history_;
    std::chrono::high_resolution_clock::time_point last_processing_time_;
    
    // Threading and synchronization
    std::atomic<bool> is_running_;
    std::atomic<bool> is_processing_;
    std::mutex input_mutex_;
    std::mutex output_mutex_;
    std::mutex metrics_mutex_;
    std::unique_ptr<std::thread> processing_thread_;
    
    // Input/Output streams
    std::unique_ptr<std::ifstream> input_file_;
    std::unique_ptr<std::ofstream> output_file_;
    std::stringstream* input_stream_;
    std::stringstream* output_stream_;

public:
    // ========================================================================
    // CONSTRUCTION AND INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Constructor
     */
    LanguageInterface();
    
    /**
     * @brief Destructor
     */
    virtual ~LanguageInterface();
    
    /**
     * @brief Initialize the language interface
     * @param config Configuration parameters
     * @return Success status
     */
    bool initialize(const std::map<std::string, float>& config = {});
    
    /**
     * @brief Shutdown the language interface
     */
    void shutdown();
    
    /**
     * @brief Set input source and processing mode
     * @param source Input source type
     * @param mode Processing mode
     * @return Success status
     */
    bool setInputSource(InputSource source, ProcessingMode mode);

    // ========================================================================
    // TEXT INPUT INTERFACE
    // ========================================================================
    
    /**
     * @brief Process text input directly
     * @param text Input text to process
     * @param context Optional context information
     * @return Processing success status
     */
    bool processTextInput(const std::string& text, const std::string& context = "");
    
    /**
     * @brief Read and process text from file
     * @param filepath Path to text file
     * @param chunk_size Size of text chunks to process
     * @return Success status
     */
    bool readTextFile(const std::string& filepath, size_t chunk_size = 1024);
    
    /**
     * @brief Process text stream continuously
     * @param input_stream Input text stream
     * @param process_realtime Whether to process in real-time
     * @return Success status
     */
    bool processTextStream(std::istream& input_stream, bool process_realtime = true);
    
    /**
     * @brief Add conversational input
     * @param user_input User's input text
     * @param conversation_context Optional conversation context
     * @return Processing success status
     */
    bool addConversationalInput(const std::string& user_input, 
                               const std::string& conversation_context = "");
    
    /**
     * @brief Get current text context
     * @return Current text being processed
     */
    std::string getCurrentTextContext() const;
    
    /**
     * @brief Get processed text segments
     * @param max_segments Maximum number of segments to return
     * @return Vector of processed text segments
     */
    std::vector<TextSegment> getProcessedSegments(size_t max_segments = 10) const;

    // ========================================================================
    // LANGUAGE ANALYSIS AND ATTENTION
    // ========================================================================
    
    /**
     * @brief Get language importance mapping
     * @return Vector of importance scores for current text
     */
    std::vector<float> getLanguageImportanceMap();
    
    /**
     * @brief Get attention guidance for language processing
     * @return Vector of attention weights for text segments
     */
    std::vector<float> getAttentionGuidance() const;
    
    /**
     * @brief Set attention focus on specific text aspects
     * @param aspect Text aspect to focus on ("keywords", "syntax", "semantics")
     * @param strength Attention strength [0.0, 1.0]
     */
    void setAttentionFocus(const std::string& aspect, float strength);
    
    /**
     * @brief Extract linguistic features from current text
     * @return Map of feature names to values
     */
    std::map<std::string, float> extractLinguisticFeatures() const;
    
    /**
     * @brief Analyze text sentiment and tone
     * @param text Text to analyze
     * @return Map of sentiment scores
     */
    std::map<std::string, float> analyzeSentiment(const std::string& text) const;
    
    /**
     * @brief Extract keywords and key phrases
     * @param text Text to analyze
     * @param max_keywords Maximum number of keywords to extract
     * @return Vector of keywords with importance scores
     */
    std::vector<std::pair<std::string, float>> extractKeywords(const std::string& text, 
                                                              size_t max_keywords = 10) const;

    // ========================================================================
    // RESPONSE GENERATION AND OUTPUT
    // ========================================================================
    
    /**
     * @brief Generate response based on current context
     * @param response_type Type of response ("answer", "question", "comment")
     * @param max_length Maximum response length
     * @return Generated response text
     */
    std::string generateResponse(const std::string& response_type = "answer", 
                               size_t max_length = 200);
    
    /**
     * @brief Output response to user/system
     * @param response Response text to output
     * @param format Output format ("plain", "formatted", "json")
     * @return Success status
     */
    bool outputResponse(const std::string& response, const std::string& format = "plain");
    
    /**
     * @brief Request more input from user/system
     * @param prompt Optional prompt to show user
     * @return Success status
     */
    bool requestMoreInput(const std::string& prompt = "");
    
    /**
     * @brief Get pending output
     * @return Queue of pending output messages
     */
    std::queue<std::string> getPendingOutput();
    
    /**
     * @brief Set output destination
     * @param filepath Path to output file (empty for console)
     * @return Success status
     */
    bool setOutputDestination(const std::string& filepath = "");

    // ========================================================================
    // CONVERSATION MANAGEMENT
    // ========================================================================
    
    /**
     * @brief Get conversation history
     * @param max_turns Maximum number of conversation turns to return
     * @return Vector of conversation turns
     */
    std::vector<ConversationTurn> getConversationHistory(size_t max_turns = 20) const;
    
    /**
     * @brief Clear conversation history
     */
    void clearConversationHistory();
    
    /**
     * @brief Set conversation context
     * @param context Context information for conversation
     */
    void setConversationContext(const std::string& context);
    
    /**
     * @brief Get current conversation context
     * @return Current conversation context
     */
    std::string getConversationContext() const;

    // ========================================================================
    // PERFORMANCE MONITORING
    // ========================================================================
    
    /**
     * @brief Get current language processing metrics
     * @return Current performance metrics
     */
    LanguageMetrics getCurrentMetrics();
    
    /**
     * @brief Get performance history
     * @return Map of metric names to historical values
     */
    std::map<std::string, float> getPerformanceHistory() const;
    
    /**
     * @brief Update processing metrics
     * @param metrics New metrics to record
     */
    void updateMetrics(const LanguageMetrics& metrics);
    
    /**
     * @brief Reset performance metrics
     */
    void resetMetrics();
    
    /**
     * @brief Get processing statistics
     * @return Human-readable statistics string
     */
    std::string getProcessingStatistics() const;

    // ========================================================================
    // CONFIGURATION AND CONTROL
    // ========================================================================
    
    /**
     * @brief Set processing parameters
     * @param parameters Map of parameter names to values
     */
    void setProcessingParameters(const std::map<std::string, float>& parameters);
    
    /**
     * @brief Get current processing parameters
     * @return Map of current parameters
     */
    std::map<std::string, float> getProcessingParameters() const;
    
    /**
     * @brief Enable or disable real-time processing
     * @param enabled Whether to enable real-time processing
     */
    void setRealTimeProcessing(bool enabled);
    
    /**
     * @brief Check if interface is currently processing
     * @return True if processing is active
     */
    bool isProcessing() const { return is_processing_.load(); }
    
    /**
     * @brief Check if interface is running
     * @return True if interface is running
     */
    bool isRunning() const { return is_running_.load(); }

private:
    // ========================================================================
    // INTERNAL HELPER METHODS
    // ========================================================================
    
    /**
     * @brief Initialize text processing components
     * @return Success status
     */
    bool initializeTextProcessing();
    
    /**
     * @brief Main processing loop for background processing
     */
    void processingLoop();
    
    /**
     * @brief Process a single text segment
     * @param segment Text segment to process
     * @return Processing success status
     */
    bool processTextSegment(const TextSegment& segment);
    
    /**
     * @brief Calculate text importance scores
     * @param text Text to analyze
     * @return Vector of importance scores
     */
    std::vector<float> calculateImportanceScores(const std::string& text);
    
    /**
     * @brief Update attention guidance based on current text
     * @param text Current text being processed
     */
    void updateAttentionGuidance(const std::string& text);
    
    /**
     * @brief Preprocess text for neural network input
     * @param text Raw input text
     * @return Preprocessed text
     */
    std::string preprocessText(const std::string& text);
    
    /**
     * @brief Tokenize text into segments
     * @param text Input text
     * @param segment_type Type of segmentation ("sentence", "paragraph")
     * @return Vector of text segments
     */
    std::vector<TextSegment> tokenizeText(const std::string& text, 
                                        const std::string& segment_type = "sentence");
    
    /**
     * @brief Calculate reading speed in words per minute
     * @param text_length Length of processed text
     * @param processing_time Time taken to process
     * @return Reading speed in WPM
     */
    float calculateReadingSpeed(size_t text_length, 
                              std::chrono::duration<float> processing_time);
    
    /**
     * @brief Validate input text
     * @param text Text to validate
     * @return True if text is valid for processing
     */
    bool validateInputText(const std::string& text);
    
    /**
     * @brief Log processing event
     * @param event Event description
     * @param level Log level ("info", "warning", "error")
     */
    void logEvent(const std::string& event, const std::string& level = "info");
};

#endif // LANGUAGE_INTERFACE_H