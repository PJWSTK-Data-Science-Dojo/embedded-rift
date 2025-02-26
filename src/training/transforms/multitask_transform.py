from .mask_frames import MaskFramesTransform
from .mask_values import MaskValuesTransform
from .next_frame_prediction import NextFramePredictionTransform
from .outcome_prediction import OutcomePredictionTransform


class MultitaskTransform:
    def __init__(
        self,
        next_frame_transform: NextFramePredictionTransform,
        mask_values_transform: MaskValuesTransform,
        mask_frames_transform: MaskFramesTransform,
        outcome_transform: OutcomePredictionTransform,
    ):
        self.next_frame_transform = next_frame_transform
        self.mask_values_transform = mask_values_transform
        self.mask_frames_transform = mask_frames_transform
        self.outcome_transform = outcome_transform

    def __call__(self, sample: dict) -> dict:
        sample = self.next_frame_transform(sample)
        sample = self.mask_values_transform(sample)
        sample = self.mask_frames_transform(sample)
        sample = self.outcome_transform(sample)
        return sample
