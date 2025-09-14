"""
nuScenes-SR Dataset Dataloader

This module provides a convenient dataloader for the nuScenes-SR dataset,
which contains scene recognition labels for autonomous driving scenarios.
"""

import json
import os
from typing import Dict, List, Tuple, Optional, Union
from collections import Counter
import random


class NuScenesSRDataloader:
    """
    Dataloader for nuScenes-SR (Scene Recognition) dataset.
    
    This class provides methods to load, filter, and iterate through 
    the scene recognition data based on nuScenes dataset.
    """
    
    # Define all available scene types
    DYNAMIC_SCENES = [
        "PED_CROSSING",      # 行人穿越
        "LEFT_TURN",         # 左转场景
        "RIGHT_TURN",        # 右转场景
        "CONSTRUCTION_VEHICLE",  # 施工车辆
        "AVOID_STATIONARY"   # 避让静止车辆
    ]
    
    STATIC_SCENES = [
        "INTERSECTION",      # 交叉路口
        "PARKING_LOT",       # 停车场
        "TRAFFIC_LIGHT",     # 交通信号灯
        "RAINY_WEATHER",     # 雨天天气
        "CONSTRUCTION_ZONE"  # 施工区域
    ]
    
    ALL_SCENES = DYNAMIC_SCENES + STATIC_SCENES
    
    def __init__(self, data_path: str = "dataset/merged_final_labels_reviewed.json"):
        """
        Initialize the dataloader.
        
        Args:
            data_path (str): Path to the JSON annotation file
        """
        self.data_path = data_path
        self.data = self._load_data()
        self.scene_tokens = list(self.data.keys())
        
    def _load_data(self) -> Dict:
        """Load the annotation data from JSON file."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} scenes from {self.data_path}")
        return data
    
    def get_scene_data(self, scene_token: str) -> Optional[Dict]:
        """
        Get data for a specific scene.
        
        Args:
            scene_token (str): Scene token from nuScenes dataset
            
        Returns:
            Dict: Scene data containing labels and description, or None if not found
        """
        return self.data.get(scene_token)
    
    def get_scenes_by_label(self, label: str) -> List[Tuple[str, Dict]]:
        """
        Get all scenes that contain a specific label.
        
        Args:
            label (str): Scene label (e.g., "PED_CROSSING")
            
        Returns:
            List[Tuple[str, Dict]]: List of (scene_token, scene_data) tuples
        """
        if label not in self.ALL_SCENES:
            raise ValueError(f"Invalid label: {label}. Must be one of {self.ALL_SCENES}")
        
        scenes = []
        for token, data in self.data.items():
            if label in data["labels"]:
                scenes.append((token, data))
        
        return scenes
    
    def get_scenes_by_labels(self, labels: List[str], mode: str = "any") -> List[Tuple[str, Dict]]:
        """
        Get scenes that contain specific labels.
        
        Args:
            labels (List[str]): List of scene labels
            mode (str): "any" (contains any of the labels) or "all" (contains all labels)
            
        Returns:
            List[Tuple[str, Dict]]: List of (scene_token, scene_data) tuples
        """
        for label in labels:
            if label not in self.ALL_SCENES:
                raise ValueError(f"Invalid label: {label}. Must be one of {self.ALL_SCENES}")
        
        scenes = []
        for token, data in self.data.items():
            scene_labels = set(data["labels"])
            target_labels = set(labels)
            
            if mode == "any" and scene_labels.intersection(target_labels):
                scenes.append((token, data))
            elif mode == "all" and target_labels.issubset(scene_labels):
                scenes.append((token, data))
        
        return scenes
    
    def get_dynamic_scenes(self) -> List[Tuple[str, Dict]]:
        """Get all scenes containing dynamic scene types."""
        return self.get_scenes_by_labels(self.DYNAMIC_SCENES, mode="any")
    
    def get_static_scenes(self) -> List[Tuple[str, Dict]]:
        """Get all scenes containing static scene types."""
        return self.get_scenes_by_labels(self.STATIC_SCENES, mode="any")
    
    def get_label_statistics(self) -> Dict[str, int]:
        """
        Get statistics of label occurrences.
        
        Returns:
            Dict[str, int]: Dictionary with label counts
        """
        label_counts = Counter()
        for data in self.data.values():
            for label in data["labels"]:
                label_counts[label] += 1
        
        return dict(label_counts)
    
    def get_multi_label_statistics(self) -> Dict[int, int]:
        """
        Get statistics of number of labels per scene.
        
        Returns:
            Dict[int, int]: Dictionary with {num_labels: count}
        """
        multi_label_counts = Counter()
        for data in self.data.values():
            num_labels = len(data["labels"])
            multi_label_counts[num_labels] += 1
        
        return dict(multi_label_counts)
    
    def sample_scenes(self, n: int = 10, random_seed: Optional[int] = None) -> List[Tuple[str, Dict]]:
        """
        Randomly sample n scenes from the dataset.
        
        Args:
            n (int): Number of scenes to sample
            random_seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            List[Tuple[str, Dict]]: List of (scene_token, scene_data) tuples
        """
        if random_seed is not None:
            random.seed(random_seed)
        
        sample_tokens = random.sample(self.scene_tokens, min(n, len(self.scene_tokens)))
        return [(token, self.data[token]) for token in sample_tokens]
    
    def split_dataset(self, train_ratio: float = 0.8, val_ratio: float = 0.1, 
                     test_ratio: float = 0.1, random_seed: Optional[int] = None) -> Tuple[List[str], List[str], List[str]]:
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            train_ratio (float): Ratio for training set
            val_ratio (float): Ratio for validation set
            test_ratio (float): Ratio for test set
            random_seed (Optional[int]): Random seed for reproducibility
            
        Returns:
            Tuple[List[str], List[str], List[str]]: (train_tokens, val_tokens, test_tokens)
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")
        
        if random_seed is not None:
            random.seed(random_seed)
        
        shuffled_tokens = self.scene_tokens.copy()
        random.shuffle(shuffled_tokens)
        
        total_scenes = len(shuffled_tokens)
        train_end = int(total_scenes * train_ratio)
        val_end = train_end + int(total_scenes * val_ratio)
        
        train_tokens = shuffled_tokens[:train_end]
        val_tokens = shuffled_tokens[train_end:val_end]
        test_tokens = shuffled_tokens[val_end:]
        
        return train_tokens, val_tokens, test_tokens
    
    def export_labels_only(self, output_path: str, scene_tokens: Optional[List[str]] = None):
        """
        Export only the labels for specified scenes (useful for ML training).
        
        Args:
            output_path (str): Output file path
            scene_tokens (Optional[List[str]]): Scene tokens to export, or None for all
        """
        if scene_tokens is None:
            scene_tokens = self.scene_tokens
        
        labels_only = {}
        for token in scene_tokens:
            if token in self.data:
                labels_only[token] = self.data[token]["labels"]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labels_only, f, indent=2, ensure_ascii=False)
        
        print(f"Exported labels for {len(labels_only)} scenes to {output_path}")
    
    def print_statistics(self):
        """Print comprehensive statistics about the dataset."""
        print("=" * 60)
        print("nuScenes-SR Dataset Statistics")
        print("=" * 60)
        
        # Basic statistics
        total_scenes = len(self.data)
        print(f"Total scenes: {total_scenes}")
        
        # Label statistics
        label_stats = self.get_label_statistics()
        print(f"\nLabel occurrences:")
        
        print("\nDynamic Scenes:")
        for label in self.DYNAMIC_SCENES:
            count = label_stats.get(label, 0)
            percentage = (count / total_scenes) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        print("\nStatic Scenes:")
        for label in self.STATIC_SCENES:
            count = label_stats.get(label, 0)
            percentage = (count / total_scenes) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        # Multi-label statistics
        multi_label_stats = self.get_multi_label_statistics()
        print(f"\nMulti-label distribution:")
        for num_labels in sorted(multi_label_stats.keys()):
            count = multi_label_stats[num_labels]
            percentage = (count / total_scenes) * 100
            print(f"  {num_labels} labels: {count} scenes ({percentage:.1f}%)")
        
        print("=" * 60)
    
    def __len__(self) -> int:
        """Return the number of scenes in the dataset."""
        return len(self.data)
    
    def __getitem__(self, scene_token: str) -> Dict:
        """Get scene data by token."""
        return self.get_scene_data(scene_token)
    
    def __iter__(self):
        """Iterate over all scenes."""
        for token in self.scene_tokens:
            yield token, self.data[token]


# Example usage and utility functions
def main():
    """Example usage of the NuScenesSRDataloader."""
    
    # Initialize dataloader
    loader = NuScenesSRDataloader()
    
    # Print dataset statistics
    loader.print_statistics()
    
    # Example: Get all pedestrian crossing scenes
    ped_crossing_scenes = loader.get_scenes_by_label("PED_CROSSING")
    print(f"\nFound {len(ped_crossing_scenes)} pedestrian crossing scenes")
    
    # Example: Get scenes with both intersection and traffic light
    intersection_light_scenes = loader.get_scenes_by_labels(
        ["INTERSECTION", "TRAFFIC_LIGHT"], mode="all"
    )
    print(f"Found {len(intersection_light_scenes)} scenes with both intersection and traffic light")
    
    # Example: Sample some scenes
    sample_scenes = loader.sample_scenes(n=5, random_seed=42)
    print(f"\nSample scenes:")
    for token, data in sample_scenes:
        print(f"  {token}: {data['labels']} - {data['description'][:50]}...")
    
    # Example: Split dataset
    train_tokens, val_tokens, test_tokens = loader.split_dataset(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )
    print(f"\nDataset split: Train={len(train_tokens)}, Val={len(val_tokens)}, Test={len(test_tokens)}")


if __name__ == "__main__":
    main()
