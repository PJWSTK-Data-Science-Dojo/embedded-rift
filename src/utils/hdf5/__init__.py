import h5py
from pathlib import Path
from typing import Any, Dict, List, Self, Tuple, Union
import numpy as np

class HDF5Database:
    """
    A class to handle reading and writing game data to multiple HDF5 files.
    """
    def __init__(self, db_dir: Union[str, Path] = "data/db", compression: Union[str, None] = "gzip") -> None:
        """
        Initialize the HDF5Database with a directory for files and a compression type.

        :param db_dir: Directory path to store database files.
        :param compression: Compression algorithm to use (e.g., "gzip").
        """
        self.db_dir: Path = Path(db_dir) if not isinstance(db_dir, Path) else db_dir
        self.games_file: Union[h5py.File, None] = None
        self.frames_file: Union[h5py.File, None] = None
        self.champions_file: Union[h5py.File, None] = None
        self.team_objectives_file: Union[h5py.File, None] = None
        self.player_timeline_file: Union[h5py.File, None] = None
        self.items_file: Union[h5py.File, None] = None
        self.games_group: Union[h5py.Group, None] = None
        self.compression: Union[str, None] = compression
        self.is_open: bool = False
    
    def get(self, collection: str, game_id: str) -> h5py.Dataset:
        """
        Retrieve dataset and metadata for a given game_id from a specified collection.

        :param collection: The collection name to read from (e.g. "games", "frames", "champions",
                        "objectives", "players", "norm_params").
        :param game_id: Unique identifier for the game (or for players, use the unique game_player_id).
        :return: A dictionary with keys 'data' (the dataset) and 'metadata' (its attributes).
        :raises ValueError: If the file is not open or the collection/game_id does not exist.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")

        file_mapping: Dict[str, h5py.File] = {
            "games": self.games_file,          # type: ignore
            "frames": self.frames_file,        # type: ignore
            "champions": self.champions_file,  # type: ignore
            "objectives": self.team_objectives_file,  # type: ignore
        }

        if collection not in file_mapping:
            raise ValueError(f"Invalid collection: {collection}.")

        file = file_mapping[collection]
        if file is None:
            raise ValueError(f"The file for collection {collection} is not open.")

        if collection not in file:
            raise ValueError(f"Collection {collection} not found in file.")

        group = file[collection]
        if game_id not in group:
            raise ValueError(f"Game {game_id} not found in collection {collection}.")

        return group[game_id]
    
    def get_game(self, game_id: str) -> h5py.Dataset:
        """
        Retrieve the game dataset.

        :param game_id: Unique game identifier.
        :return: h5py.Dataset for the game data.
        """
        return self.games_group[game_id]

    def get_frames(self, game_id: str) -> h5py.Dataset:
        """
        Retrieve the frames dataset for a given game.

        :param game_id: Unique game identifier.
        :return: h5py.Dataset for the frames data.
        """
        return self.get("frames", game_id)

    def get_champions(self, game_id: str) -> h5py.Dataset:
        """
        Retrieve the champions dataset for a given game.

        :param game_id: Unique game identifier.
        :return: h5py.Dataset for the champions data.
        """
        return self.get("champions", game_id)

    def get_team_objectives(self, game_id: str) -> h5py.Dataset:
        """
        Retrieve the team objectives dataset for a given game.

        :param game_id: Unique game identifier.
        :return: h5py.Dataset for the team objectives data.
        """
        return self.get("objectives", game_id)

    def get_player_timeline(self, game_id: str, player_idx: int) -> h5py.Dataset:
        """
        Retrieve the player timeline dataset.

        :param game_id: Unique game identifier.
        :param player_idx: Index of the player (should be used with the game_player_id convention).
        :return: h5py.Dataset for the player timeline data.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        gpg = self._get_player_group(player_idx)
        return gpg[game_id]
    
    def get_items(self, game_id: str) -> h5py.Dataset:
        """
        Retrieve the items dataset for a given game from the items file.

        :param game_id: Unique game identifier.
        :return: h5py.Dataset for the items data.
        :raises ValueError: If the file is not open or the game_id is not found.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")

        if "items" not in self.items_file:
            raise ValueError("Items group not found in items file.")

        group = self.items_file["items"]
        if game_id not in group:
            raise ValueError(f"Items for game {game_id} not found in the items group.")
        
        return group[game_id]
    
    def get_game_player_entries(self) -> List[str]:
        """
        Retrieve a list of pseudo IDs for game-player timeline entries.
        Each pseudo ID is formatted as "game_id_playerIdx", where the game_id is taken from
        the dataset key and the player_idx is read from the dataset attributes.
        
        :return: List of pseudo IDs for each player timeline entry.
        :raises ValueError: If the database is not open or the 'players' group is missing.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        
        entires = []
        players_grp = self.player_timeline_file.get("players", None)
        if players_grp is None:
            raise ValueError("No 'players' group found in the player timeline file.")
        
        # Iterate over each subgroup inside "players" (e.g., "group_0", "group_1", ..., "group_9")
        for subgroup_key in players_grp.keys():
            subgroup = players_grp[subgroup_key]
            idx = int(subgroup_key[-1])
            for game_id in subgroup.keys():
                entires.append((game_id, idx))
                
        return entires

    def get_player_timeline_by_pseudo_id(self, pseudo_id: str) -> h5py.Dataset:
        """
        Retrieve the player timeline dataset using a pseudo ID formatted as "game_id_playerIdx".

        :param pseudo_id: Pseudo ID string (e.g., "12345_3"), where the last part is the player index.
        :return: h5py.Dataset corresponding to the player timeline.
        :raises ValueError: If the pseudo ID is not formatted correctly or the dataset is not found.
        """
        game_id, player_idx = self._pseudo_id_to_game_player_idx(pseudo_id)

        # Use the existing method to retrieve the player timeline dataset.
        return self.get_player_timeline(game_id, player_idx)
    
    def get_player_items(self, game_id: str, player_idx: int) -> np.ndarray:
        """
        Retrieve the items for a specific player using a pseudo ID formatted as "game_id_playerIdx".
        The items are stored as a dataset under the "items" group in the items_file with shape 
        (num_frames, num_players, num_items). This method returns only the items for the given player.
        
        :param pseudo_id: Pseudo ID string (e.g., "12345_3") where the last part is the player index.
        :return: A numpy array containing the player's items with shape (num_frames, num_items).
        :raises ValueError: If the pseudo ID is not formatted correctly or the dataset is not found.
        """
        # Parse the pseudo ID. We assume that the last part is the player index,
        # and the rest (joined back together) is the game ID.

        
        # Retrieve the full items dataset for the game.
        items_ds = self.get_items(game_id)
        items_arr = np.array(items_ds)  # Expected shape: (num_frames, num_players, num_items)
        
        # Ensure the player index is within bounds.
        if player_idx < 0 or player_idx >= items_arr.shape[1]:
            raise ValueError(f"Player index {player_idx} out of bounds for game {game_id}.")
        
        # Extract and return only the items corresponding to this player.
        player_items = items_arr[:, player_idx, :]  # shape: (num_frames, num_items)
        return player_items
    
    def get_timeline_norms(self, group: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_open:
            raise ValueError("No database file is open.")
        
        
        norms_ds = self.player_timeline_file["norm_params"]
        
        if group is None:
            norms = norms_ds["global"]
        else:
            norms = norms_ds[f"group_{group}"]
            
        mean = norms["means"][:]
        std = norms["std"][:]
        
        return mean, std
        
    
    def remove_game(self, game_id: str):
        """
        Remove all datasets associated with a game from the database.

        :param game_id: Unique game identifier.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        
        # self.games_file["games"].__delitem__(game_id)
        # self.frames_file["frames"].__delitem__(game_id)
        # self.champions_file["champions"].__delitem__(game_id)
        # self.team_objectives_file["objectives"].__delitem__(game_id)
        # self.items_file["items"].__delitem__(game_id)
        for player_idx in range(10):
            gpg = self._get_player_group(player_idx)
            gpg.__delitem__(game_id)
        
    @staticmethod
    def _pseudo_id_to_game_player_idx(pseudo_id: str) -> Tuple[str, int]:
        """
        Convert a pseudo ID to a game ID and player index.

        :param pseudo_id: Pseudo ID string (e.g., "12345_3") where the last part is the player index.
        :return: Tuple of game ID and player index.
        :raises ValueError: If the pseudo ID is not formatted correctly.
        """
        try:
            parts = pseudo_id.split('_')
            if len(parts) < 2:
                raise ValueError("Pseudo ID must be formatted as 'game_id_playerIdx'.")
            player_idx = int(parts[-1])
            game_id = '_'.join(parts[:-1])
            return game_id, player_idx
        except Exception as e:
            raise ValueError(f"Error parsing pseudo ID '{pseudo_id}': {e}")
    
    def __len__(self) -> int:
        """
        Return the number of games in the database.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        return len(self.games_group)  # type: ignore
    
    def open(self) -> None:
        """
        Open all database files and create groups if they do not exist.
        """
        self.db_dir.mkdir(parents=True, exist_ok=True)
        
        self.games_file = h5py.File(self.db_dir / "games.h5", "a")
        if "games" not in self.games_file:
            self.games_file.create_group("games")
            
        self.games_group = self.games_file["games"]
        
        self.frames_file = h5py.File(self.db_dir / "frames.h5", "a")
        self.champions_file = h5py.File(self.db_dir / "champions.h5", "a")
        self.team_objectives_file = h5py.File(self.db_dir / "team_objectives.h5", "a")
        self.player_timeline_file = h5py.File(self.db_dir / "players_frames.h5", "a")
        self.items_file = h5py.File(self.db_dir / "items.h5", "a")
        self.is_open = True
                
    def close(self) -> None:
        """
        Close all open HDF5 files.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        
        self.games_file.close()  # type: ignore
        self.frames_file.close()  # type: ignore
        self.champions_file.close()  # type: ignore
        self.team_objectives_file.close()  # type: ignore
        self.player_timeline_file.close()  # type: ignore
        self.items_file.close()  # type: ignore
        self.is_open = False
        
    def write_game(self, game_id: str, game_data: Dict[str, Any]) -> List[Union[float, int]]:
        """
        Write game-level data into the games file.

        :param game_id: Unique identifier for the game.
        :param game_data: Dictionary containing game data.
        :return: List containing game duration, early surrender, surrender, and blue win status.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        
        collection = "games"
        games_group: h5py.Group = self._check_group(self.games_file, collection, game_id)
        
        game_duration = game_data["game_duration"]
        early_surrender = int(game_data["early_surrender"])
        surrender = int(game_data["surrender"])
        blue_win = int(game_data["blue_win"])
        data: List[Union[float, int]] = [game_duration, early_surrender, surrender, blue_win]
        
        ds = games_group.create_dataset(game_id, data=data, compression=self.compression)
        self._add_metadata(ds, game_data)
        return data
        
    def write_frame(self, game_id: str, game_data: Dict[str, Any]) -> np.ndarray:
        """
        Write frame-level data for a game into the frames file.

        :param game_id: Unique game identifier.
        :param game_data: Dictionary containing frame data.
        :return: Numpy array of frame data.
        """
        self._check_in_db(game_id)
        collection = "frames"
        frames_group: h5py.Group = self._check_group(self.frames_file, collection, game_id)
        
        frames = game_data["frames"]
        first_frame = frames[0]
        keys_order = list(first_frame.keys())
        frame_vectors: List[List[float]] = []
        for frame in frames:
            vector = [float(frame.get(key, 0)) for key in keys_order]
            frame_vectors.append(vector)
        frame_array = np.array(frame_vectors, dtype=np.float32)
        
        frame_ds = frames_group.create_dataset(game_id, data=frame_array, compression=self.compression)
        self._add_metadata(frame_ds, game_data)
        return frame_array
    
    def write_champions(self, game_id: str, game_data: Dict[str, Any]) -> np.ndarray:
        """
        Write champion data for a game into the champions file.

        :param game_id: Unique game identifier.
        :param game_data: Dictionary containing champion data.
        :return: Numpy array with blue and red champions.
        """
        self._check_in_db(game_id)
        collection = "champions"
        champions_group: h5py.Group = self._check_group(self.champions_file, collection, game_id)
        blue_champions = game_data["blue_champions"]
        red_champions = game_data["red_champions"]
        
        ds = champions_group.create_dataset(
            game_id,
            data=[blue_champions, red_champions],
            dtype=np.int32,
            compression=self.compression,
        )
        self._add_metadata(ds, game_data)
        return np.array([blue_champions, red_champions], dtype=np.int32)
    
    def write_team_objectives(self, game_id: str, game_data: Dict[str, Any]) -> np.ndarray:
        """
        Write team objectives data into the team objectives file.

        :param game_id: Unique game identifier.
        :param game_data: Dictionary containing objectives data.
        :return: Numpy array of objectives data.
        """
        self._check_in_db(game_id)
        collection = "objectives"
        objectives_group: h5py.Group = self._check_group(self.team_objectives_file, collection, game_id)
        
        objectives_data = game_data["objectives"]
        objectives_vectors: List[List[int]] = []
        for frame in objectives_data:
            vector = [int(team.get(key, 0)) for team in frame for key in team]
            objectives_vectors.append(vector)
        objectives_array = np.array(objectives_vectors, dtype=np.int32)
        
        ds = objectives_group.create_dataset(game_id, data=objectives_array, compression=self.compression)
        self._add_metadata(ds, game_data)
        return objectives_array
    
    def write_player_timeline(self, game_id: str, player_idx: int, players_data: Dict[str, Any], game_data: Dict[str, Any]) -> np.ndarray:
        """
        Write timeline data for a specific player into the players file.

        :param game_id: Unique game identifier.
        :param player_idx: Index of the player.
        :param players_data: Dictionary containing timeline data for players.
        :param game_data: Dictionary containing game data.
        :return: Numpy array of player timeline data.
        """
        self._check_in_db(game_id)

        players_group: h5py.Group = self._check_player_group(game_id, player_idx)        
        all_champions = game_data["blue_champions"] + game_data["red_champions"]
        
        champion_id = all_champions[player_idx]
        player_data = players_data[player_idx]
        
        player_vectors: List[List[float]] = []
        for frame in player_data:
            vector = [float(frame.get(key, 0)) for key in frame]
            player_vectors.append(vector)
        player_array = np.array(player_vectors, dtype=np.float32)
        
        ds = players_group.create_dataset(game_id, data=player_array, compression=self.compression)
        ds.attrs["player_champion"] = champion_id
        ds.attrs["game_id"] = game_id
        ds.attrs["player_idx"] = player_idx
        ds.attrs["team"] = "blue" if player_idx < 5 else "red"
        self._add_metadata(ds, game_data)
        return player_array
    
    def write_items(self, game_id: str, game_data: Dict[str, Any]) -> np.ndarray:
        """
        Write item data for a game into the items file.

        :param game_id: Unique game identifier.
        :param game_data: Dictionary containing item data.
        :return: Numpy array of item data.
        """
        self._check_in_db(game_id)
        collection = "items"
        items_group: h5py.Group = self._check_group(self.items_file, collection, game_id)
        
        items_data = game_data["items_per_frame"]
        items_vectors: List[List[int]] = []
        for frame in items_data:
            vector = [[int(item) for item in player] for player in frame]
            items_vectors.append(vector)
        items_array = np.array(items_vectors, dtype=np.int32)
        
        ds = items_group.create_dataset(game_id, data=items_array, compression=self.compression)
        self._add_metadata(ds, game_data)
        return items_array
    
    def _add_metadata(self, ds: h5py.Dataset, metadata: Dict[str, Any]) -> None:
        """
        Add common metadata attributes to a dataset.

        :param ds: HDF5 dataset to update.
        :param metadata: Dictionary containing metadata.
        """
        ds.attrs["platform"] = metadata["platform"]
        ds.attrs["season"] = metadata["season"]
        ds.attrs["patch"] = metadata["patch"]
        ds.attrs["game_duration"] = metadata["game_duration"]
        ds.attrs["early_surrender"] = metadata["early_surrender"]
        ds.attrs["surrender"] = metadata["surrender"]
        ds.attrs["blue_win"] = metadata["blue_win"]
        
    def _check_in_db(self, game_id: str) -> None:
        """
        Check if a game exists in the database.

        :param game_id: Unique game identifier.
        :raises ValueError: If the game does not exist.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        if game_id not in self.games_group:  # type: ignore
            raise ValueError(f"Game {game_id} does not exist in the database.")
        
    
    @staticmethod
    def _check_group(file: h5py.File, collection: str, game_id: str) -> h5py.Group:
        """
        Check for a collection group in a file and create it if it doesn't exist.
        Also, ensure the game_id is not already present.

        :param file: HDF5 file.
        :param collection: Group name within the file.
        :param game_id: Unique game identifier.
        :return: HDF5 group for the collection.
        :raises ValueError: If the game_id already exists.
        """
        if collection not in file:
            file.create_group(collection)
        group: h5py.Group = file[collection]
        if game_id in group:
            raise ValueError(f"Frames for game {game_id} already exist in the database.")
        return group

    def _get_player_group(self, player_idx: int) -> h5py.Group:
        players_group: h5py.Group = self.player_timeline_file["players"]
        gpg = players_group[f"group_{player_idx}"]
        return gpg

    def _check_player_group(self, game_id: str, player_idx: int) -> h5py.Group:
        if "players" not in self.player_timeline_file:
            players_group = self.player_timeline_file.create_group("players")
            for idx in range(10):
                player_key = f"group_{idx}"
                players_group.create_group(player_key)
            
        gpg = self._get_player_group(player_idx)
        if game_id in gpg:
            raise ValueError(f"Player {player_idx} data for game {game_id} already exists in the database.")
        
        return gpg        
    
    def __contains__(self, game_id: str) -> bool:
        """
        Check if a game exists in the database.

        :param game_id: Unique game identifier.
        :return: True if exists, False otherwise.
        """
        if not self.is_open:
            raise ValueError("No database file is open.")
        return game_id in self.games_group  # type: ignore
    
    def __enter__(self) -> Self:
        """
        Enter the runtime context and open the database.
        """
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Exit the runtime context and close the database.
        """
        self.close()
        return False
