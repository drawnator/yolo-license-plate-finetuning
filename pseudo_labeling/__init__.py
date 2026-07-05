"""Semi-supervised pseudo-labeling pipeline for the unified YOLO label space.

This package completes missing object-class annotations across the project's YOLO
datasets by training class-specialized teacher models, generating pseudo-labels on
missing-class images, and merging accepted detections without destroying ground truth.
"""
