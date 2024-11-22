"""3D ML pipelines for torch."""

from .multitask import Multitask
from .semantic_segmentation import SemanticSegmentation
from .object_detection import ObjectDetection

__all__ = ['Multitask', 'SemanticSegmentation', 'ObjectDetection']
