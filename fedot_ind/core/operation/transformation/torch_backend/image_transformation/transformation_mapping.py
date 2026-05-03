from fedot_ind.core.operation.transformation.torch_backend.image_transformation.methods.mtf_transformation import MTF
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.methods.gaf_transformation import GAF
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.methods.stft_transformation import STFTSpectrogram
from fedot_ind.core.operation.transformation.torch_backend.image_transformation.types import ImageTransformationType


TRANSFORMATION_MAPPING = {
    ImageTransformationType.MTF: MTF,
    ImageTransformationType.GAF: GAF,
    ImageTransformationType.STFT: STFTSpectrogram,
}