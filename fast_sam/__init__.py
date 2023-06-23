from .fast_sam import FastSAM
from .fast_sam import FastSamAutomaticMaskGenerator

fast_sam_model_registry = {
    "FastSAM-x": FastSAM,
    "FastSAM-s": FastSAM,
}
