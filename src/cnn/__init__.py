"""CNN Visual Feature Extraction Package for EduPulse AI.

Provides modular visual feature extraction from tracked student crops using
a pretrained CNN backbone (ResNet18). Preserves spatial-temporal metadata
for downstream temporal sequence modelling (RNN/LSTM/GRU).
"""

from src.cnn.feature_extractor import (
    CNNFeatureExtractor,
    ExtractionSummary,
    run_cnn_feature_extraction,
)
from src.cnn.visualization import (
    PCA_DISCLAIMER_TEXT,
    PCA_DISCLAIMER_TITLE,
    compute_pca_2d,
    create_pca_scatter_figure,
)

__all__ = [
    "CNNFeatureExtractor",
    "ExtractionSummary",
    "run_cnn_feature_extraction",
    "compute_pca_2d",
    "create_pca_scatter_figure",
    "PCA_DISCLAIMER_TITLE",
    "PCA_DISCLAIMER_TEXT",
]
