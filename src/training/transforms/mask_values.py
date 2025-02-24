import torch


class MaskValuesTransform:
    """
    Masks individual feature values in each frame.
    For each value in the frame tensor, it is masked (set to zero)
    with a probability of mask_val_prob.
    """

    def __init__(self, mask_val_prob: float = 0.15):
        self.mask_val_prob = mask_val_prob

    def __call__(self, sample: dict) -> dict:
        frames = sample["frames"]
        # Create a mask for individual values.
        mask = torch.rand(frames.shape) < self.mask_val_prob
        frames_masked_values = frames.clone()
        frames_masked_values[mask] = (
            0.0  # You could also replace with a special token value if needed.
        )
        sample["mask_values"] = mask
        sample["frames_masked_values"] = frames_masked_values
        return sample
