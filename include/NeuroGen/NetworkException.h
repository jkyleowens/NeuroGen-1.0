#ifndef NETWORK_EXCEPTION_H
#define NETWORK_EXCEPTION_H

#include <stdexcept>
#include <string>

/**
 * @enum NetworkError
 * @brief Defines the types of errors that can occur within the neural network simulation.
 */
enum class NetworkError {
    UNKNOWN,
    CONFIGURATION_ERROR,
    INITIALIZATION_FAILED,
    NETWORK_NOT_INITIALIZED,
    INVALID_INPUT,
    CUDA_ERROR,
    FILE_NOT_FOUND,
    MODEL_NOT_SUPPORTED
};

/**
 * @class NetworkException
 * @brief Custom exception class for the NeuroGen simulation.
 *
 * This class provides a structured way to handle errors, combining an
 * error code with a descriptive message.
 */
class NetworkException : public std::runtime_error {
public:
    /**
     * @brief Constructs a NetworkException.
     * @param error_code The type of error from the NetworkError enum.
     * @param message A detailed message describing the specific error.
     */
    NetworkException(NetworkError error_code, const std::string& message)
        : std::runtime_error(format_message(error_code, message)),
          m_error_code(error_code) {}

    /**
     * @brief Returns the error code associated with the exception.
     * @return The NetworkError enum value.
     */
    NetworkError get_error_code() const {
        return m_error_code;
    }

private:
    NetworkError m_error_code;

    /**
     * @brief Formats the error message for the what() method.
     * @param error_code The error code.
     * @param message The detailed error message.
     * @return A formatted string combining the error code and message.
     */
    static std::string format_message(NetworkError error_code, const std::string& message) {
        return "Network Error [" + error_code_to_string(error_code) + "]: " + message;
    }

    /**
     * @brief Converts a NetworkError enum value to its string representation.
     * @param error_code The error code to convert.
     * @return A string name for the error code.
     */
    static std::string error_code_to_string(NetworkError error_code) {
        switch (error_code) {
            case NetworkError::CONFIGURATION_ERROR:     return "Configuration Error";
            case NetworkError::INITIALIZATION_FAILED:   return "Initialization Failed";
            case NetworkError::NETWORK_NOT_INITIALIZED: return "Network Not Initialized";
            case NetworkError::INVALID_INPUT:           return "Invalid Input";
            case NetworkError::CUDA_ERROR:              return "CUDA Error";
            case NetworkError::FILE_NOT_FOUND:          return "File Not Found";
            case NetworkError::MODEL_NOT_SUPPORTED:     return "Model Not Supported";
            default:                                    return "Unknown Error";
        }
    }
};

#endif // NETWORK_EXCEPTION_H