import numpy as np

class PlayerTimelineTransform:
    def __init__(self, mask_frame_prob: float = 0.0):
        """
        Args:
            mask_frame_prob: Probability of masking an entire frame.
            concat_team_objectives: If True, concatenate team objectives as the first features.
        """
        self.mask_frame_prob = mask_frame_prob

    def get_ally_enemy_champions(self, champions: np.ndarray, side: int) -> tuple:
        """
        Args:
            champions: Array of champion IDs.
            side: 0 for blue, 1 for red.
        
        Returns:
            Tuple of ally and enemy champions.
        """
        blue_red_ids = champions.reshape(2, 5)
        blue_champions = blue_red_ids[0]
        red_champions = blue_red_ids[1]
        ally_champions = blue_champions if not side else red_champions
        enemy_champions = red_champions if side else blue_champions
        return ally_champions, enemy_champions

    def mask_frames(self, frames: np.ndarray, mask_prob: float) -> np.ndarray:
        """
        Args:
            frames: Array of frames.
            mask_prob: Probability of masking a frame.
        
        Returns:
            Masked frames.
        """
        seq_len, _ = frames.shape
        
        mask = np.random.rand(seq_len) < mask_prob
        frames[mask, :] = 0
        
        return frames
    
    def add_team_objectives(self, frames: np.ndarray, team_objectives: np.ndarray) -> np.ndarray:
        """
        Args:
            frames: Array of frames.
            team_objectives: Array of team objectives.
        
        Returns:
            Frames with team objectives concatenated.
        """        
        # Concatenate team objectives to the beginning of each frame's features.
        return np.concatenate([team_objectives, frames], axis=1)
    
    def __call__(self, sample: dict) -> dict:
        # Get the original frames from the sample.
        frames = self.add_team_objectives(sample["frames"].copy(), sample["team_objectives"])
        
        champions = sample["champions"]  # shape: (seq_len, 10)
        side = sample["side"]  # 0 for blue, 1 for red
        
        ally_champions, enemy_champions = self.get_ally_enemy_champions(champions, side)
                
        masked_frames = self.mask_frames(frames, self.mask_frame_prob)  # Apply masking to the frames.
        
        sample["original_frames"] = frames
        sample["frames"] = masked_frames
        sample["ally_champions"] = ally_champions
        sample["enemy_champions"] = enemy_champions
        
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
        "side": 0,  # 0 for blue
        "champions": np.random.randint(0, 100, size=(10, )),  # Dummy champions

    }
    
    transform = PlayerTimelineTransform(mask_frame_prob=0.2, concat_team_objectives=True)
    transformed_sample = transform(sample)
    
    print("Original frames shape:", transformed_sample["original_frames"].shape)  # Expected: (10, 2*obj_dim + feature_dim)
    print("Masked frames shape:", transformed_sample["frames"].shape)             # Expected: (10, 2*obj_dim + feature_dim)
    print("Updated norm_means shape:", transformed_sample["norm_means"].shape)      # Expected: (10, 2*obj_dim + feature_dim)
    print("Updated norm_stds shape:", transformed_sample["norm_stds"].shape)        # Expected: (10, 2*obj_dim + feature_dim)
