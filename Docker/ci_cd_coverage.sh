#!/bin/bash
# Quick coverage generation for both Python and C++ packages

# ============================================
# PACKAGE CONFIGURATION
# ============================================
# Define your ROS2 packages here
CPP_PACKAGES=()
PY_PACKAGES=("face_recognition" "face_detection" "visual_speech_activity" "gaze_estimation")
# ============================================

set -e
cd /workspace

echo "=========================================="
echo "  Code Coverage Report Generation"
echo "=========================================="
echo ""

# Python coverage
for PY_PKG in "${PY_PACKAGES[@]}"; do
  echo "📊 Python Coverage ($PY_PKG)"
  echo "------------------------------------------"
  PY_OUTPUT=$(colcon test --packages-select $PY_PKG --event-handlers console_direct+ --pytest-args --cov=$PY_PKG --cov-report=term --cov-report=html --cov-report=lcov 2>&1)
  echo "$PY_OUTPUT" | grep -B1 -A 10 "coverage: platform" | grep -v "^--$"
  echo ""
done
# C++ coverage
for CPP_PKG in "${CPP_PACKAGES[@]}"; do
  echo "📊 C++ Coverage ($CPP_PKG)"
  echo "------------------------------------------"
  
  if [ ! -d "build/$CPP_PKG" ]; then
    echo "⚠️  Package build directory not found: build/$CPP_PKG"
    echo ""
    continue
  fi
  
  cd build/$CPP_PKG

# Check if coverage data exists
if ! find . -name "*.gcda" -type f | grep -q .; then
    echo "⚠️  No coverage data found!"
    echo "   C++ package needs to be built with coverage flags."
    echo ""
    echo "   To enable C++ coverage, rebuild with:"
    echo "   colcon build --packages-select $CPP_PKG --cmake-clean-cache \\"
    echo "     --cmake-args -DCMAKE_CXX_FLAGS='--coverage' \\"
    echo "                  -DCMAKE_C_FLAGS='--coverage' \\"
    echo "                  -DCMAKE_EXE_LINKER_FLAGS='--coverage'"
    echo ""
    echo "   Then run tests: colcon test --packages-select $CPP_PKG"
    cd /workspace
else
    # Generate coverage report
    lcov --capture --directory . --output-file coverage.info --ignore-errors mismatch,inconsistent >/dev/null 2>&1
    lcov --remove coverage.info '/usr/*' '*/test/*' '*/build/*' '*/gtest/*' '*/rosidl_*' --output-file coverage_filtered.info --ignore-errors unused >/dev/null 2>&1
    genhtml coverage_filtered.info --output-directory coverage_html --quiet 2>/dev/null
    
    echo "Project Source Files:"
    lcov --list coverage_filtered.info 2>/dev/null | grep -E "workspace/src/$CPP_PKG" -A1 | grep -E "(\.cpp|\.hpp)" | sed 's/^/  /'
    cd /workspace
  fi
done

echo ""
echo "=========================================="
echo "✅ Coverage Reports Generated!"
echo "=========================================="
echo ""
echo "📁 HTML Reports:"
for PY_PKG in "${PY_PACKAGES[@]}"; do
  if [ -d "src/$PY_PKG/htmlcov" ]; then
    echo "   Python ($PY_PKG):  src/$PY_PKG/htmlcov/index.html"
  fi
done
for CPP_PKG in "${CPP_PACKAGES[@]}"; do
  if [ -f "/workspace/build/$CPP_PKG/coverage_html/index.html" ]; then
    echo "   C++ ($CPP_PKG):     build/$CPP_PKG/coverage_html/index.html"
  fi
done
echo ""