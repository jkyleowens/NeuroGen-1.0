#!/bin/bash

# ============================================================================
# BRAIN MODULE ARCHITECTURE PERFORMANCE EVALUATION SCRIPT
# File: run_performance_evaluation.sh
# ============================================================================

echo "=== BRAIN MODULE ARCHITECTURE PERFORMANCE EVALUATION ==="
echo "Starting comprehensive test suite and performance analysis..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CUDA is available
check_cuda() {
    print_status "Checking CUDA availability..."
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        print_success "CUDA found: version $CUDA_VERSION"
        return 0
    else
        print_warning "CUDA not found - will run CPU-only tests"
        return 1
    fi
}

# Build the project
build_project() {
    print_status "Building project..."
    
    # Clean previous builds
    make clean > /dev/null 2>&1
    
    # Build main project
    if make -j$(nproc) > build.log 2>&1; then
        print_success "Main project built successfully"
    else
        print_error "Failed to build main project"
        echo "Build log:"
        cat build.log
        return 1
    fi
    
    # Build test suite
    if make test_brain_architecture > test_build.log 2>&1; then
        print_success "Test suite built successfully"
    else
        print_error "Failed to build test suite"
        echo "Test build log:"
        cat test_build.log
        return 1
    fi
    
    return 0
}

# Run the comprehensive test suite
run_test_suite() {
    print_status "Running comprehensive test suite..."
    
    if [ -f "./test_brain_architecture" ]; then
        echo ""
        echo "=========================================="
        echo "EXECUTING BRAIN ARCHITECTURE TEST SUITE"
        echo "=========================================="
        echo ""
        
        # Run tests and capture output
        ./test_brain_architecture | tee test_results.log
        
        # Check if tests passed
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            print_success "Test suite completed"
        else
            print_error "Some tests failed"
            return 1
        fi
    else
        print_error "Test executable not found"
        return 1
    fi
    
    return 0
}

# Analyze test results
analyze_results() {
    print_status "Analyzing test results..."
    
    if [ -f "test_results.log" ]; then
        echo ""
        echo "=========================================="
        echo "TEST RESULTS ANALYSIS"
        echo "=========================================="
        
        # Extract key metrics
        TOTAL_TESTS=$(grep "Total Tests:" test_results.log | awk '{print $3}')
        PASSED_TESTS=$(grep "Passed:" test_results.log | awk '{print $6}')
        FAILED_TESTS=$(grep "Failed:" test_results.log | awk '{print $9}')
        SUCCESS_RATE=$(grep "Success Rate:" test_results.log | awk '{print $3}')
        TOTAL_TIME=$(grep "Total Time:" test_results.log | awk '{print $12}')
        
        echo "📊 Test Summary:"
        echo "   Total Tests: $TOTAL_TESTS"
        echo "   Passed: $PASSED_TESTS"
        echo "   Failed: $FAILED_TESTS"
        echo "   Success Rate: $SUCCESS_RATE"
        echo "   Total Execution Time: $TOTAL_TIME"
        echo ""
        
        # Performance metrics
        echo "⚡ Performance Metrics:"
        if grep -q "Processing speed:" test_results.log; then
            PROCESSING_SPEED=$(grep "Processing speed:" test_results.log | awk '{print $3}')
            FPS=$(grep "Processing speed:" test_results.log | awk '{print $6}' | tr -d '()')
            echo "   Processing Speed: $PROCESSING_SPEED per frame"
            echo "   Frame Rate: $FPS"
        fi
        
        # Memory usage
        if grep -q "Memory usage test:" test_results.log; then
            MEMORY_INSTANCES=$(grep "Memory usage test:" test_results.log | awk '{print $5}')
            echo "   Memory Test: Successfully created $MEMORY_INSTANCES brain instances"
        fi
        
        echo ""
        
        # Check for critical failures
        if [ "$FAILED_TESTS" != "0" ]; then
            print_warning "Some tests failed - check test_results.log for details"
            echo ""
            echo "Failed Tests:"
            grep "FAIL" test_results.log | while read line; do
                echo "   ❌ $line"
            done
        else
            print_success "All tests passed! 🎉"
        fi
        
    else
        print_error "Test results file not found"
        return 1
    fi
    
    return 0
}

# Generate performance report
generate_report() {
    print_status "Generating performance report..."
    
    REPORT_FILE="performance_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$REPORT_FILE" << EOF
# Brain Module Architecture Performance Report

**Generated:** $(date)
**System:** $(uname -a)
**CPU:** $(lscpu | grep "Model name" | cut -d: -f2 | xargs)
**Memory:** $(free -h | grep "Mem:" | awk '{print $2}')

## Test Results Summary

EOF

    if [ -f "test_results.log" ]; then
        echo "### Overall Results" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        grep -A 10 "TEST SUITE SUMMARY" test_results.log >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        echo "### Individual Test Results" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        grep -E "(Running Test:|Result:)" test_results.log >> "$REPORT_FILE"
        echo "\`\`\`" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        
        echo "### Performance Benchmarks" >> "$REPORT_FILE"
        echo "" >> "$REPORT_FILE"
        if grep -q "Processing speed:" test_results.log; then
            echo "- **Processing Speed:** $(grep "Processing speed:" test_results.log | awk '{print $3, $4, $5, $6, $7}')" >> "$REPORT_FILE"
        fi
        if grep -q "Memory usage test:" test_results.log; then
            echo "- **Memory Usage:** $(grep "Memory usage test:" test_results.log)" >> "$REPORT_FILE"
        fi
        echo "" >> "$REPORT_FILE"
    fi
    
    echo "### Architecture Details" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo "- **Modules:** 9 specialized brain modules" >> "$REPORT_FILE"
    echo "- **Visual Input:** 1920x1080 resolution support" >> "$REPORT_FILE"
    echo "- **Neural Networks:** Individual networks per module" >> "$REPORT_FILE"
    echo "- **Learning:** STDP, homeostasis, attention-based modulation" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    
    print_success "Performance report generated: $REPORT_FILE"
}

# Main execution
main() {
    echo "Starting performance evaluation at $(date)"
    echo ""
    
    # Check prerequisites
    check_cuda
    CUDA_AVAILABLE=$?
    
    # Build project
    if ! build_project; then
        print_error "Build failed - aborting evaluation"
        exit 1
    fi
    
    # Run tests
    if ! run_test_suite; then
        print_error "Test suite failed"
        exit 1
    fi
    
    # Analyze results
    analyze_results
    
    # Generate report
    generate_report
    
    echo ""
    print_success "Performance evaluation completed!"
    echo ""
    echo "📁 Generated files:"
    echo "   - test_results.log (detailed test output)"
    echo "   - performance_report_*.md (summary report)"
    echo "   - build.log (build output)"
    echo ""
    echo "🚀 Next steps:"
    echo "   - Review test results for any failures"
    echo "   - Check performance metrics against requirements"
    echo "   - Run './test_brain_architecture' for detailed output"
    echo ""
}

# Execute main function
main "$@"
