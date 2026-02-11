#!/bin/bash
# Local verification script for CI/CD pipeline
# Mirrors the GitHub Actions workflow for build, test, and coverage

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

print_help() {
    cat << EOF
${BLUE}Usage:${NC} $0 [OPTIONS]

${YELLOW}Description:${NC}
  Local CI/CD pipeline verification script for ROS 2 packages.
  Builds, tests, and generates coverage reports for specified packages.

${YELLOW}Options:${NC}
  -c, --cpp PACKAGE       C++ package name(s) (comma-separated)
  -p, --python PACKAGE    Python package name(s) (comma-separated)
  -a, --all              Process all packages in workspace
  --clean                Clean build artifacts before starting
  --no-clean             Skip clean prompt (keep artifacts)
  -h, --help             Display this help message

${YELLOW}Examples:${NC}
  # Test specific packages
  $0 --cpp audio_stream_manager --python speech_recognition

  # Test only C++ package
  $0 --cpp audio_stream_manager

  # Test only Python package
  $0 --python speech_recognition

  # Test all packages with clean build
  $0 --all --clean

${YELLOW}Package Type Detection:${NC}
  - C++ packages: Built with coverage flags (--coverage)
  - Python packages: Built normally, tested with pytest-cov
  - If package type not specified, script attempts auto-detection

${YELLOW}Output:${NC}
  - Test results: test_summary.txt
  - C++ coverage: build/PACKAGE/coverage_html/index.html
  - Python coverage: PACKAGE/htmlcov/index.html
  - Cobertura XML: coverage-cpp.xml, coverage-py.xml

EOF
    exit 0
}

# Parse command line arguments
CPP_PACKAGES=()
PY_PACKAGES=()
ALL_PACKAGES=false
CLEAN_BUILD=""
AUTO_CLEAN=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cpp)
            IFS=',' read -ra ADDR <<< "$2"
            for pkg in "${ADDR[@]}"; do
                CPP_PACKAGES+=("$pkg")
            done
            shift 2
            ;;
        -p|--python)
            IFS=',' read -ra ADDR <<< "$2"
            for pkg in "${ADDR[@]}"; do
                PY_PACKAGES+=("$pkg")
            done
            shift 2
            ;;
        -a|--all)
            ALL_PACKAGES=true
            shift
            ;;
        --clean)
            CLEAN_BUILD="yes"
            AUTO_CLEAN=false
            shift
            ;;
        --no-clean)
            CLEAN_BUILD="no"
            AUTO_CLEAN=false
            shift
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo -e "${RED}Error: Unknown option '$1'${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect packages if --all is specified
if [ "$ALL_PACKAGES" = true ]; then
    if [ -d "src" ]; then
        for pkg_dir in src/*; do
            if [ -d "$pkg_dir" ]; then
                pkg_name=$(basename "$pkg_dir")
                # Check if it's a C++ package (has CMakeLists.txt)
                if [ -f "$pkg_dir/CMakeLists.txt" ]; then
                    # Check if it has Python code too
                    if [ -d "$pkg_dir/$pkg_name" ] && [ -n "$(find "$pkg_dir/$pkg_name" -name '*.py' 2>/dev/null)" ]; then
                        CPP_PACKAGES+=("$pkg_name")
                        PY_PACKAGES+=("$pkg_name")
                    else
                        CPP_PACKAGES+=("$pkg_name")
                    fi
                # Pure Python package (has setup.py or setup.cfg)
                elif [ -f "$pkg_dir/setup.py" ] || [ -f "$pkg_dir/setup.cfg" ]; then
                    PY_PACKAGES+=("$pkg_name")
                fi
            fi
        done
    fi
fi

# Validate that we have at least one package
if [ ${#CPP_PACKAGES[@]} -eq 0 ] && [ ${#PY_PACKAGES[@]} -eq 0 ]; then
    echo -e "${RED}Error: No packages specified${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Change to workspace root
cd /workspace

print_header "🚀 Local CI/CD Pipeline Verification"
print_info "C++ Packages: ${CPP_PACKAGES[*]:-none}"
print_info "Python Packages: ${PY_PACKAGES[*]:-none}"

# Step 1: Source ROS environment
print_header "📦 Step 1: Setup ROS Environment"
if [ -f "/opt/ros/jazzy/setup.bash" ]; then
    source /opt/ros/jazzy/setup.bash
    print_success "ROS 2 Jazzy environment sourced"
else
    print_error "ROS 2 Jazzy not found at /opt/ros/jazzy"
    exit 1
fi

# Step 2: Clean previous build artifacts (optional)
print_header "🧹 Step 2: Clean Previous Build (Optional)"
if [ "$AUTO_CLEAN" = true ]; then
    read -p "Clean previous build/install/log directories? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        CLEAN_BUILD="yes"
    else
        CLEAN_BUILD="no"
    fi
fi

if [ "$CLEAN_BUILD" = "yes" ]; then
    rm -rf build install log
    print_success "Cleaned previous build artifacts"
else
    print_info "Keeping previous build artifacts"
fi

set -e

# Step 3: Build dependencies first
print_header "🔨 Step 3: Build Dependencies"
print_info "Building all dependencies first to ensure proper environment..."

# Always build common dependencies that most packages need
COMMON_DEPS=("hri_msgs")
for dep in "${COMMON_DEPS[@]}"; do
    if [ -d "src/$dep" ]; then
        print_info "Building dependency: $dep"
        colcon build --symlink-install --packages-select "$dep"
        if [ $? -ne 0 ]; then
            print_error "Failed to build dependency: $dep"
            exit 1
        fi
    fi
done

# Step 4: Build C++ packages with coverage flags
if [ ${#CPP_PACKAGES[@]} -gt 0 ]; then
    print_header "🔨 Step 4: Build C++ Packages with Coverage"
    for pkg in "${CPP_PACKAGES[@]}"; do
        print_info "Building C++ package: $pkg"
        colcon build --symlink-install \
            --packages-up-to "$pkg" \
            --cmake-args \
                -DCMAKE_CXX_FLAGS='--coverage' \
                -DCMAKE_C_FLAGS='--coverage' \
                -DCMAKE_EXE_LINKER_FLAGS='--coverage'
        
        if [ $? -eq 0 ]; then
            print_success "$pkg built successfully with coverage instrumentation"
        else
            print_error "$pkg build failed"
            exit 1
        fi
    done
else
    print_info "No C++ packages to build"
fi

# Step 5: Build Python packages
if [ ${#PY_PACKAGES[@]} -gt 0 ]; then
    print_header "🔨 Step 5: Build Python Packages"
    # Build Python packages (excluding C++ ones already built)
    PY_ONLY_PACKAGES=()
    for pkg in "${PY_PACKAGES[@]}"; do
        # Check if package is not in CPP_PACKAGES
        if [[ ! " ${CPP_PACKAGES[@]} " =~ " ${pkg} " ]]; then
            PY_ONLY_PACKAGES+=("$pkg")
        fi
    done
    
    if [ ${#PY_ONLY_PACKAGES[@]} -gt 0 ]; then
        for pkg in "${PY_ONLY_PACKAGES[@]}"; do
            print_info "Building Python package: $pkg (with dependencies)"
            colcon build --symlink-install --packages-up-to "$pkg"
            
            if [ $? -eq 0 ]; then
                print_success "$pkg built successfully"
            else
                print_error "$pkg build failed"
                exit 1
            fi
        done
    else
        print_info "All Python packages already built as part of C++ packages"
    fi
else
    print_info "No Python packages to build"
fi

# Step 6: Run tests
print_header "🧪 Step 6: Run All Tests"
set +e  # Don't exit on test failures

ALL_PKG_LIST=()
for pkg in "${CPP_PACKAGES[@]}"; do
    ALL_PKG_LIST+=("$pkg")
done
for pkg in "${PY_PACKAGES[@]}"; do
    if [[ ! " ${ALL_PKG_LIST[@]} " =~ " ${pkg} " ]]; then
        ALL_PKG_LIST+=("$pkg")
    fi
done

if [ ${#ALL_PKG_LIST[@]} -gt 0 ]; then
    # Set up environment variables for different packages
    export PYTHONDONTWRITEBYTECODE=1  # Prevent .pyc files during testing
    
    # For EutHRIFaces packages, ensure AI environment is available
    # All packages (face_detection, face_recognition, gaze_estimation, visual_speech_activity) use AI_VENV
    if [ ${#PY_PACKAGES[@]} -gt 0 ]; then
        if [ -d "/opt/ros_python_env" ]; then
            export AI_VENV="/opt/ros_python_env"
            print_info "Using ros_python_env for AI-based packages tests"
        else
            print_info "AI environment not found at /opt/ros_python_env, tests will use mocks"
        fi
    fi
    
    # Build pytest args with coverage for Python packages
    # Note: pytest.ini already includes "-v --tb=short" in addopts
    PYTEST_ARGS_ARRAY=()
    if [ ${#PY_PACKAGES[@]} -gt 0 ]; then
        # Add coverage for Python packages (pytest.ini handles -v and --tb=short)
        for pkg in "${PY_PACKAGES[@]}"; do
            PYTEST_ARGS_ARRAY+=("--cov=$pkg")
        done
        # Memory-optimized settings to prevent OOM (Error 137)
        PYTEST_ARGS_ARRAY+=("--cov-report=html" "--cov-report=term" "--cov-report=xml")
        PYTEST_ARGS_ARRAY+=("-p" "no:cacheprovider")  # Disable caching to reduce memory
    fi
    
    # Convert array to proper string for colcon
    if [ ${#PYTEST_ARGS_ARRAY[@]} -gt 0 ]; then
        # Join array elements with spaces for proper parsing
        PYTEST_ARGS_STR=$(printf "%s " "${PYTEST_ARGS_ARRAY[@]}")
        PYTEST_ARGS_STR=${PYTEST_ARGS_STR% }  # Remove trailing space
        print_info "Running tests with additional pytest args: $PYTEST_ARGS_STR"
        # Use the coverage args (pytest.ini will add -v --tb=short automatically)
        colcon test \
            --packages-select "${ALL_PKG_LIST[@]}" \
            --event-handlers console_direct+ \
            --pytest-args "$PYTEST_ARGS_STR" \
            --return-code-on-test-failure
        TEST_EXIT_CODE=$?
    else
        print_info "Running tests with default pytest configuration (from pytest.ini)"
        # No additional args needed - pytest.ini handles everything
        colcon test \
            --packages-select "${ALL_PKG_LIST[@]}" \
            --event-handlers console_direct+ \
            --return-code-on-test-failure
        TEST_EXIT_CODE=$?
    fi
else
    print_info "Running tests for all packages with default pytest configuration"
    colcon test \
        --event-handlers console_direct+ \
        --return-code-on-test-failure
    TEST_EXIT_CODE=$?
fi

set -e

if [ $TEST_EXIT_CODE -eq 0 ]; then
    print_success "All tests passed"
else
    print_error "Some tests failed (exit code: $TEST_EXIT_CODE)"
fi

# Step 7: Generate test summary
print_header "📊 Step 7: Generate Test Summary"
colcon test-result --all --verbose > test_summary.txt
cat test_summary.txt

# Parse test results
if [ -f test_summary.txt ]; then
    SUMMARY_LINE=$(grep "^Summary:" test_summary.txt || echo "")
    if [ -n "$SUMMARY_LINE" ]; then
        print_success "Test summary generated"
        echo -e "${BLUE}$SUMMARY_LINE${NC}"
        
        # Extract test statistics
        TOTAL=$(echo "$SUMMARY_LINE" | grep -oP '\d+(?= tests)' || echo "0")
        FAILURES=$(echo "$SUMMARY_LINE" | grep -oP '\d+(?= failures)' || echo "0")
        ERRORS=$(echo "$SUMMARY_LINE" | grep -oP '\d+(?= errors)' || echo "0")
        
        PASSED=$((TOTAL - FAILURES - ERRORS))
        FAILED=$((FAILURES + ERRORS))
        
        echo -e "${GREEN}Passed: $PASSED${NC} | ${RED}Failed: $FAILED${NC} | Total: $TOTAL"
    fi
fi

# Step 8: Generate coverage reports
print_header "📈 Step 8: Generate Coverage Reports"

# Python coverage
for pkg in "${PY_PACKAGES[@]}"; do
    print_info "Generating Python coverage for: $pkg"
    
    # Check for coverage data in build directory (colcon puts it there)
    build_dir="build/$pkg"
    if [ -d "$build_dir" ]; then
        cd "$build_dir"
        
        # Look for pytest coverage data
        if [ -f ".coverage" ] || find . -name ".coverage*" -type f | grep -q .; then
            # Generate HTML coverage report
            python3 -c "
import sys
sys.path.insert(0, '/workspace/src/$pkg')
try:
    import coverage
    cov = coverage.Coverage()
    cov.load()
    cov.html_report(directory='/workspace/${pkg}_htmlcov')
    cov.report(file=open('/workspace/${pkg}_coverage.txt', 'w'))
    print('Coverage generated for $pkg')
except Exception as e:
    print(f'Coverage generation failed: {e}')
" 2>/dev/null || echo "No coverage data to process"
            print_success "$pkg Python coverage generated"
        else
            print_info "No Python coverage data found for $pkg"
        fi
        cd /workspace
    else
        print_info "Build directory not found for $pkg"
    fi
done

# C++ coverage
for pkg in "${CPP_PACKAGES[@]}"; do
    print_info "Generating C++ coverage for: $pkg"
    if [ -d "build/$pkg" ]; then
        cd "build/$pkg"
        
        # Check if we have gcov data (search recursively)
        if find . -name "*.gcda" -type f | grep -q .; then
            # Generate initial coverage info
            lcov --capture --directory . --output-file coverage.info --ignore-errors mismatch,inconsistent 2>/dev/null || true
            
            # Filter out system, test, and external libraries
            lcov --remove coverage.info \
                '/usr/*' \
                '*/test/*' \
                '*/install/*' \
                '*/build/*' \
                '*/gtest/*' \
                '*/rosidl_*' \
                --output-file coverage_filtered.info --ignore-errors unused 2>/dev/null || true
            
            # Generate HTML report
            genhtml coverage_filtered.info --output-directory coverage_html --quiet 2>/dev/null || true
            
            print_success "$pkg C++ coverage generated"
        else
            print_info "No C++ coverage data found for $pkg (no .gcda files)"
        fi
        cd /workspace
    fi
done

# Step 9: Display coverage summary
print_header "📊 Step 9: Coverage Summary"

# Python coverage summaries
for pkg in "${PY_PACKAGES[@]}"; do
    echo -e "\n${BLUE}=== Python Coverage: $pkg ===${NC}"
    if [ -f "${pkg}_coverage.txt" ]; then
        cat "${pkg}_coverage.txt"
    else
        print_info "Python coverage report not found for $pkg"
    fi
done

# C++ coverage summaries
for pkg in "${CPP_PACKAGES[@]}"; do
    echo -e "\n${BLUE}=== C++ Coverage: $pkg ===${NC}"
    if [ -f "build/$pkg/coverage_filtered.info" ]; then
        lcov --summary "build/$pkg/coverage_filtered.info" 2>&1 | grep -E "lines|functions"
    else
        print_info "C++ coverage report not found for $pkg"
    fi
done

# Step 10: Summary
print_header "✨ Pipeline Verification Complete!"

echo -e "${BLUE}Generated Artifacts:${NC}"
echo "  📄 Test Results:"
[ -f "test_summary.txt" ] && echo "    - test_summary.txt" || echo "    - test_summary.txt (not found)"

for pkg in "${CPP_PACKAGES[@]}" "${PY_PACKAGES[@]}"; do
    [ -d "build/$pkg/test_results" ] && echo "    - build/$pkg/test_results/" || true
done

echo -e "\n  📊 Coverage Reports (LCOV/Info):"
for pkg in "${CPP_PACKAGES[@]}"; do
    [ -f "build/$pkg/coverage_filtered.info" ] && echo "    - build/$pkg/coverage_filtered.info" || true
done
for pkg in "${PY_PACKAGES[@]}"; do
    [ -f "build/$pkg/coverage.lcov" ] && echo "    - build/$pkg/coverage.lcov" || true
done

echo -e "\n  🌐 HTML Coverage Reports:"
for pkg in "${CPP_PACKAGES[@]}"; do
    if [ -d "build/$pkg/coverage_html" ]; then
        HTML_PATH="/workspace/build/$pkg/coverage_html/index.html"
        echo -e "    ${GREEN}C++ ($pkg):${NC} $HTML_PATH"
    fi
done
for pkg in "${PY_PACKAGES[@]}"; do
    if [ -d "$pkg/htmlcov" ]; then
        HTML_PATH="/workspace/$pkg/htmlcov/index.html"
        echo -e "    ${GREEN}Python ($pkg):${NC} $HTML_PATH"
    fi
done

echo -e "\n${GREEN}🎉 Local verification complete!${NC}"
echo -e "${YELLOW}💡 You can now safely push to trigger the GitHub Actions workflow${NC}"
echo ""
echo -e "${BLUE}To view HTML reports in a terminal/docker environment:${NC}"
echo -e "  Use a file browser or mount the workspace volume to access HTML files"
echo -e "  Or use: ${YELLOW}python3 -m http.server 8000${NC} and open browser to http://localhost:8000"
echo ""

# Exit with test exit code
exit $TEST_EXIT_CODE