import torch


class NextFramePredictionTransform:
    """
    Creates a target for next frame prediction by shifting the frames.
    For a sequence of T frames, this transform sets the target as frames 1..T-1.
    """

    def __call__(self, sample: dict) -> dict:
        frames = sample["frames"]
        # Ensure frames is a torch tensor
        if not torch.is_tensor(frames):
            frames = torch.tensor(frames, dtype=torch.float32)
            sample["frames"] = frames

        # Create target sequence by shifting by one time step.
        if frames.shape[0] > 1:
            sample["next_frames"] = frames[1:].clone()
        else:
            sample["next_frames"] = None

        return sample
