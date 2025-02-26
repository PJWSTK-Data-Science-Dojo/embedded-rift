import torch


class OutcomePredictionTransform:
    """
    Converts the 'blue_win' flag in the sample into a scalar tensor outcome.
    """

    def __call__(self, sample: dict) -> dict:
        outcome = sample.get("blue_win", False)
        sample["outcome"] = torch.tensor(1.0 if outcome else 0.0, dtype=torch.float32)
        return sample
