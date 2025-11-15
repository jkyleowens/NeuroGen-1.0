class LanguageDatasetReader {
public:
    struct TextSample {
        std::string text;
        std::string target;  // For supervised learning
        std::vector<int> token_ids;
        float difficulty_score = 1.0f;
    };
    
    LanguageDatasetReader() = default;
    
    bool loadDataset(const std::string& dataset_path);
    bool hasNextBatch() const;
    std::vector<TextSample> getNextBatch(int batch_size);
    void reset();
    size_t getDatasetSize() const { return samples_.size(); }
    
private:
    std::vector<TextSample> samples_;
    size_t current_index_ = 0;
    std::string dataset_path_;
};