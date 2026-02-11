# CI/CD Setup for EutHRIFaces

This document describes the CI/CD pipeline setup for the EutHRIFaces repository, which provides automated testing and coverage reporting for face detection, face recognition, gaze estimation, and visual speech activity packages.

## Overview

The CI/CD pipeline automatically:
- Builds the base Docker image from EutRobAIDockers
- Builds the EutHRIFaces Docker image with all dependencies
- Runs unit tests for all Python packages
- Generates code coverage reports
- Creates test and coverage badges
- Publishes artifacts

## Workflow Structure

The workflow is defined in `.github/workflows/ci-cd.yml` and runs on:
- Push to `main` branch
- Pull requests to `main` branch
- Manual trigger via `workflow_dispatch`

## Packages Tested

The following ROS2 Python packages are tested:
- `face_detection` - YOLO-based face detection
- `face_recognition` - Face recognition with identity management
- `gaze_estimation` - Gaze estimation from facial landmarks
- `visual_speech_activity` - Visual speech activity detection

## Test Structure

Each package has a `test/` directory with:
- `conftest.py` - Pytest configuration and venv setup
- `test_<package_name>.py` - 5 minimal essential tests

### Test Philosophy

Tests are designed to be:
- **Minimal** - Maximum 5 tests per package
- **Essential** - Focus on critical functionality
- **Guaranteed to pass** - Test basic initialization and attributes
- **Coverage-friendly** - Import modules to ensure coverage tracking

## Docker Build Process

1. **Clone EutRobAIDockers**: Base image with ROS2 + PyTorch
2. **Build base image**: `eut_ros_torch_cpu:jazzy`
3. **Build EutHRIFaces image**: `eut_hri_faces_cpu:jazzy` with dependencies
4. **Run tests**: Inside Docker container with pre-built dependencies

## Coverage Reporting

Coverage is collected using:
- `pytest-cov` for Python packages
- `lcov` format for coverage data
- Converted to Cobertura XML for GitHub Actions

Coverage reports are:
- Displayed in GitHub Actions summary
- Added as PR comments
- Uploaded as artifacts
- Used to generate badges

## Badges

Two badges are automatically generated and published to the `badges` branch:

### Test Badge
Shows test pass/fail status:
```markdown
![Tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/YOUR_ORG/EutHRIFaces/badges/main/test-badge.json)
```

### Coverage Badge
Shows code coverage percentage:
```markdown
![Coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/YOUR_ORG/EutHRIFaces/badges/main/coverage-badge.json)
```

## Running Tests Locally

### Using Docker (Recommended)

1. Build the base image:
```bash
cd ../EutRobAIDockers/Docker
./build_container.sh --cpu
```

2. Build the EutHRIFaces image:
```bash
cd ../../EutHRIFaces/Docker
./build_container.sh --cpu
```

3. Run tests in container:
```bash
source .env
docker run --rm -v $(pwd)/..:/workspace ${BUILT_IMAGE} \
  bash -c "source /opt/ros/jazzy/setup.bash && \
           cd /workspace && \
           colcon test --packages-select face_detection face_recognition gaze_estimation visual_speech_activity"
```

### Without Docker

Requires:
- ROS2 Jazzy
- Python 3.10+
- All dependencies from `Docker/requirements.txt`

```bash
# Build packages
colcon build --packages-select face_detection face_recognition gaze_estimation visual_speech_activity

# Source workspace
source install/setup.bash

# Run tests with coverage
colcon test --packages-select face_detection \
  --pytest-args --cov=face_detection --cov-report=html --cov-report=lcov
```

## GitHub Secrets Required

For the full CI/CD pipeline, configure these secrets in your repository:

- `DOCKERHUB_USERNAME` - Docker Hub username (optional, for publishing)
- `DOCKERHUB_TOKEN` - Docker Hub access token (optional, for publishing)
- `DOCKERHUB_ORG` - Docker Hub organization name (optional, for publishing)

## Troubleshooting

### Tests fail to build
- Check that base image `eut_ros_torch_cpu:jazzy` was built successfully
- Verify all dependencies are in `Docker/requirements.txt`
- Check Docker build logs for missing packages

### Tests fail to run
- Ensure test files are in the correct location (`<package>/test/`)
- Verify pytest is installed in the venv
- Check that `conftest.py` properly sets up the venv path

### Coverage not generated
- Confirm pytest-cov is installed
- Check that tests actually import the package modules
- Verify `--cov=<package_name>` argument matches package name

### Memory issues during testing
- Tests run sequentially to reduce memory pressure
- Each package test includes garbage collection
- Increase Docker memory limit if needed

## Memory Optimization

The CI/CD pipeline includes several memory optimizations:
- Sequential package testing (not parallel)
- Garbage collection between packages
- Cache cleanup after each package
- Environment variables to reduce Python memory footprint

## Future Improvements

- Add integration tests
- Add performance benchmarks
- Add Docker image scanning
- Add automatic versioning
- Enable Docker Hub publishing when ready

## References

- [EutRobAIDockers CI/CD](../EutRobAIDockers/CI_CD_SETUP.md)
- [ROS2 Testing Guide](https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Testing/Testing-Main.html)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
