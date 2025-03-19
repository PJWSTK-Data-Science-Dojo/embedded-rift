import numpy as np

class PlayerTimelineTransform:
    def __init__(self, mask_frame_prob: float = 0.0, concat_team_objectives: bool = True):
        """
        Args:
            mask_frame_prob: Probability of masking an entire frame.
            concat_team_objectives: If True, concatenate team objectives as the first features.
        """
        self.mask_frame_prob = mask_frame_prob
        self.concat_team_objectives = concat_team_objectives

    def __call__(self, sample: dict) -> dict:
        # Get the original frames from the sample.
        original_frames = sample["frames"].copy()  # shape: (seq_len, feature_dim)
        seq_len, _ = original_frames.shape
        
    
        if self.concat_team_objectives:
            team_objectives = sample["team_objectives"]

            # Concatenate team objectives to the beginning of each frame's features.
            original_frames = np.concatenate([team_objectives, original_frames], axis=1)
            
            # Update normalization statistics if provided.
            means = sample["norm_means"]  # shape: (seq_len, feature_dim)
            stds = sample["norm_stds"]    # shape: (seq_len, feature_dim)

            team_means = sample["obj_means"]
            team_stds = sample["obj_stds"]
            # Concatenate along the feature dimension.
            sample["norm_means"] = np.concatenate([team_means, means], axis=1)
            sample["norm_stds"] = np.concatenate([team_stds, stds], axis=1)
        
        masked_frames = original_frames.copy()       # copy to apply masking
        
        # Apply frame masking: for each frame, with probability mask_frame_prob, set the entire frame to 0.
        if self.mask_frame_prob > 0:
            mask = np.random.rand(seq_len) < self.mask_frame_prob
            masked_frames[mask, :] = 0.0
        
        # Add two fields: one for the original (unmasked) frames and one for the masked frames.
        
        sample["original_frames"] = original_frames  # Used for loss computation.
        sample["frames"] = masked_frames               # Input to the model.

        return sample

# Example usage:
if __name__ == "__main__":
    # Dummy sample for demonstration.
    seq_len = 10
    feature_dim = 30
    obj_dim = 5  # per team
    # Simulate team objectives as separate blue and red arrays: shape (seq_len, 2, obj_dim)
    team_objectives = np.random.randn(seq_len, 2, obj_dim)
    
    sample = {
        "frames": np.random.randn(seq_len, feature_dim),
        "team_objectives": team_objectives,
        "norm_means": np.random.randn(seq_len, feature_dim),
        "norm_stds": np.abs(np.random.randn(seq_len, feature_dim)) + 1e-6,  # avoid zeros
    }
    
    transform = PlayerTimelineTransform(mask_frame_prob=0.2, concat_team_objectives=True)
    transformed_sample = transform(sample)
    
    print("Original frames shape:", transformed_sample["original_frames"].shape)  # Expected: (10, 2*obj_dim + feature_dim)
    print("Masked frames shape:", transformed_sample["frames"].shape)             # Expected: (10, 2*obj_dim + feature_dim)
    print("Updated norm_means shape:", transformed_sample["norm_means"].shape)      # Expected: (10, 2*obj_dim + feature_dim)
    print("Updated norm_stds shape:", transformed_sample["norm_stds"].shape)        # Expected: (10, 2*obj_dim + feature_dim)
