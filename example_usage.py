"""
Example usage of nuScenes-SR dataloader
This script demonstrates how to use the dataloader for various tasks.
"""

from dataloader import NuScenesSRDataloader


def example_basic_usage():
    """Basic usage examples."""
    print("=" * 60)
    print("Basic Usage Examples")
    print("=" * 60)
    
    # Initialize dataloader
    loader = NuScenesSRDataloader()
    
    # Get basic information
    print(f"Total scenes: {len(loader)}")
    print(f"Available scene types: {len(loader.ALL_SCENES)}")
    
    # Get a specific scene
    first_token = loader.scene_tokens[0]
    scene_data = loader.get_scene_data(first_token)
    print(f"\nFirst scene: {first_token}")
    print(f"Labels: {scene_data['labels']}")
    print(f"Description: {scene_data['description']}")


def example_scene_filtering():
    """Examples of filtering scenes by labels."""
    print("\n" + "=" * 60)
    print("Scene Filtering Examples")
    print("=" * 60)
    
    loader = NuScenesSRDataloader()
    
    # Find pedestrian crossing scenes
    ped_scenes = loader.get_scenes_by_label("PED_CROSSING")
    print(f"Pedestrian crossing scenes: {len(ped_scenes)}")
    
    # Find complex intersection scenes (with traffic lights)
    intersection_scenes = loader.get_scenes_by_labels(
        ["INTERSECTION", "TRAFFIC_LIGHT"], mode="all"
    )
    print(f"Intersection + Traffic light scenes: {len(intersection_scenes)}")
    
    # Find any dynamic scenes
    dynamic_scenes = loader.get_dynamic_scenes()
    print(f"Dynamic scenes: {len(dynamic_scenes)}")
    
    # Find any static scenes
    static_scenes = loader.get_static_scenes()
    print(f"Static scenes: {len(static_scenes)}")


def example_statistics():
    """Examples of dataset statistics."""
    print("\n" + "=" * 60)
    print("Dataset Statistics Examples")
    print("=" * 60)
    
    loader = NuScenesSRDataloader()
    
    # Get label statistics
    label_stats = loader.get_label_statistics()
    print("Top 5 most common labels:")
    sorted_labels = sorted(label_stats.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_labels[:5]:
        print(f"  {label}: {count}")
    
    # Get multi-label statistics
    multi_label_stats = loader.get_multi_label_statistics()
    print(f"\nMulti-label distribution:")
    for num_labels in sorted(multi_label_stats.keys()):
        count = multi_label_stats[num_labels]
        print(f"  {num_labels} labels: {count} scenes")


def example_data_splitting():
    """Examples of dataset splitting."""
    print("\n" + "=" * 60)
    print("Dataset Splitting Examples")
    print("=" * 60)
    
    loader = NuScenesSRDataloader()
    
    # Split dataset for machine learning
    train_tokens, val_tokens, test_tokens = loader.split_dataset(
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42
    )
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_tokens)} scenes ({len(train_tokens)/len(loader)*100:.1f}%)")
    print(f"  Validation: {len(val_tokens)} scenes ({len(val_tokens)/len(loader)*100:.1f}%)")
    print(f"  Test: {len(test_tokens)} scenes ({len(test_tokens)/len(loader)*100:.1f}%)")
    
    # Export training labels
    loader.export_labels_only("example_train_labels.json", train_tokens[:10])  # Only first 10 for demo
    print(f"Exported sample training labels to example_train_labels.json")


def example_sampling():
    """Examples of random sampling."""
    print("\n" + "=" * 60)
    print("Random Sampling Examples")
    print("=" * 60)
    
    loader = NuScenesSRDataloader()
    
    # Sample random scenes
    sample_scenes = loader.sample_scenes(n=5, random_seed=42)
    print("Random sample of 5 scenes:")
    
    for i, (token, data) in enumerate(sample_scenes, 1):
        labels_str = ", ".join(data['labels'])
        description = data['description'][:50] + "..." if len(data['description']) > 50 else data['description']
        print(f"  {i}. Labels: [{labels_str}]")
        print(f"     Description: {description}")
        print()


def example_iteration():
    """Examples of iterating through the dataset."""
    print("\n" + "=" * 60)
    print("Dataset Iteration Examples")
    print("=" * 60)
    
    loader = NuScenesSRDataloader()
    
    # Iterate through first 3 scenes
    print("First 3 scenes in dataset:")
    count = 0
    for token, data in loader:
        if count >= 3:
            break
        labels_str = ", ".join(data['labels'])
        print(f"  {token}: [{labels_str}]")
        count += 1


def main():
    """Run all examples."""
    try:
        example_basic_usage()
        example_scene_filtering()
        example_statistics()
        example_data_splitting()
        example_sampling()
        example_iteration()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the dataset file 'dataset/merged_final_labels_reviewed.json' exists.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
