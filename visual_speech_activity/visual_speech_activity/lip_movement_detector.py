#!/usr/bin/env python3
"""
Lip Movement Detector - Enhanced Version

This module provides an enhanced lip movement detector that supports both full
dlib facial landmarks (68 points) and simple 5-point landmarks for visual
speech activity detection.

The detector uses RNN-based temporal classification and multiple lip movement
features for robust speaking detection.
"""

# Import the enhanced detector and provide backward compatibility
from .enhanced_lip_movement_detector import EnhancedLipMovementDetector

# Backward compatibility alias
LipMovementDetector = EnhancedLipMovementDetector

__all__ = ['LipMovementDetector', 'EnhancedLipMovementDetector']
