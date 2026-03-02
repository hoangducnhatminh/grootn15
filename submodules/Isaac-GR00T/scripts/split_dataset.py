import json
import os
import random
import shutil
from pathlib import Path

def split_dataset(dataset_path, train_ratio=0.8, seed=42):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist.")
        return

    random.seed(seed)

    # Load episodes
    episodes_path = dataset_path / "meta" / "episodes.jsonl"
    with open(episodes_path, "r") as f:
        episodes = [json.loads(line) for line in f]

    # Load tasks
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    with open(tasks_path, "r") as f:
        tasks = [json.loads(line) for line in f]

    # Load info
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path, "r") as f:
        info = json.load(f)

    # Shuffle and split
    random.shuffle(episodes)
    split_idx = int(len(episodes) * train_ratio)
    train_episodes = episodes[:split_idx]
    test_episodes = episodes[split_idx:]

    # Get episode indices for task filtering
    train_indices = {ep["episode_index"] for ep in train_episodes}
    test_indices = {ep["episode_index"] for ep in test_episodes}

    # Ensure tasks have episode_indices (build from episodes if missing)
    for t in tasks:
        if "episode_indices" not in t:
            # Map task description to indices from episodes.jsonl
            t["episode_indices"] = [
                ep["episode_index"] for ep in episodes 
                if t["task"] in ep.get("tasks", [])
            ]

    # Filter tasks
    train_tasks = [t for t in tasks if any(idx in train_indices for idx in t["episode_indices"])]
    test_tasks = [t for t in tasks if any(idx in test_indices for idx in t["episode_indices"])]

    # Function to create subset
    def create_subset(name, sub_episodes, sub_tasks):
        subset_path = dataset_path.parent / f"{dataset_path.name}_{name}"
        subset_path.mkdir(parents=True, exist_ok=True)
        (subset_path / "meta").mkdir(exist_ok=True)

        # Copy static meta files
        shutil.copy(dataset_path / "meta" / "modality.json", subset_path / "meta" / "modality.json")
        shutil.copy(dataset_path / "meta" / "stats.json", subset_path / "meta" / "stats.json")

        # Write filtered episodes and tasks
        with open(subset_path / "meta" / "episodes.jsonl", "w") as f:
            for ep in sorted(sub_episodes, key=lambda x: x["episode_index"]):
                f.write(json.dumps(ep) + "\n")
        
        with open(subset_path / "meta" / "tasks.jsonl", "w") as f:
            for t in sub_tasks:
                # Filter episode_indices in task to only include those in this subset
                # (Optional, depends on how much strictness is needed)
                t_subset = t.copy()
                indices_in_subset = [idx for idx in t["episode_indices"] if idx in (train_indices if name == "train" else test_indices)]
                t_subset["episode_indices"] = indices_in_subset
                f.write(json.dumps(t_subset) + "\n")

        # Write updated info.json
        sub_info = info.copy()
        sub_info["total_episodes"] = len(sub_episodes)
        sub_info["total_frames"] = sum(ep["length"] for ep in sub_episodes)
        sub_info["total_videos"] = len(sub_episodes) # Assuming 1 video per episode
        sub_info["total_tasks"] = len(sub_tasks)
        with open(subset_path / "meta" / "info.json", "w") as f:
            json.dump(sub_info, f, indent=4)

        # Copy data and videos for each episode
        for ep in sub_episodes:
            idx = ep["episode_index"]
            # Currently assuming chunk-000 as found in the dataset
            # (In a more complex setup, you'd map idx to chunk)
            chunk = "chunk-000"
            
            # Copy parquet data
            src_parquet = dataset_path / "data" / chunk / f"episode_{idx:06d}.parquet"
            dst_parquet_dir = subset_path / "data" / chunk
            dst_parquet_dir.mkdir(parents=True, exist_ok=True)
            if src_parquet.exists():
                shutil.copy(src_parquet, dst_parquet_dir / f"episode_{idx:06d}.parquet")
            
            # Copy video data (maintaining structure)
            # Based on: videos/chunk-000/observation.images.ego_view/episode_000000.mp4
            src_video_dir = dataset_path / "videos" / chunk / "observation.images.ego_view"
            src_video = src_video_dir / f"episode_{idx:06d}.mp4"
            dst_video_dir = subset_path / "videos" / chunk / "observation.images.ego_view"
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            if src_video.exists():
                shutil.copy(src_video, dst_video_dir / f"episode_{idx:06d}.mp4")

        print(f"Created {name} subset at {subset_path} with {len(sub_episodes)} episodes (copies).")

    create_subset("train", train_episodes, train_tasks)
    create_subset("test", test_episodes, test_tasks)

if __name__ == "__main__":
    split_dataset("dataset/generated_dataset_annotated_gr1")
