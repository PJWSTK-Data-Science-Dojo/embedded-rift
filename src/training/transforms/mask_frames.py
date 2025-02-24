import torch


class MaskFramesTransform:
    """
    Masks entire frames with a probability of mask_frame_prob.
    For each frame (row) in the sequence, if selected, the whole frame is zeroed out.
    """

    def __init__(self, mask_frame_prob: float = 0.1):
        self.mask_frame_prob = mask_frame_prob

    def __call__(self, sample: dict) -> dict:
        frames = sample["frames"]
        num_frames = frames.shape[0]
        # Create a mask for entire frames.
        mask = torch.rand(num_frames) < self.mask_frame_prob
        frames_masked = frames.clone()
        frames_masked[mask] = 0.0  # Zero out the entire frame for masked frames.
        sample["mask_frames"] = mask
        sample["frames_masked"] = frames_masked
        return sample
