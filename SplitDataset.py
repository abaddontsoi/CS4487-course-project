import os
import shutil
from pathlib import Path
import random
import argparse

def split_train_to_val(source_dir, val_ratio=0.2, seed=42):
    source_dir = Path(source_dir)
    val_dir = source_dir.parent / "val"
    
    random.seed(seed)
    
    # Define class subdirectories
    classes = ["0_real", "1_fake"]
    
    for class_name in classes:
        train_class_dir = source_dir / class_name
        val_class_dir = val_dir / class_name
        
        if not train_class_dir.exists():
            print(f"Warning: {train_class_dir} does not exist. Skipping.")
            continue
            
        # Create validation directory
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = [f for f in train_class_dir.iterdir() 
                if f.suffix.lower() in image_extensions and f.is_file()]
        
        if len(files) == 0:
            print(f"No images found in {train_class_dir}")
            continue
            
        # Calculate number to move
        num_to_move = max(1, int(len(files) * val_ratio))  # at least 1 image
        print(f"Moving {num_to_move}/{len(files)} images from {class_name} to validation")
        
        # Randomly select files
        files_to_move = random.sample(files, num_to_move)
        
        # Move them
        for file_path in files_to_move:
            dest_path = val_class_dir / file_path.name
            shutil.move(str(file_path), str(dest_path))
            # print(f"Moved: {file_path.name} → {dest_path}")

    print(f"\nDone! Validation set created at: {val_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move 20% of train data to val/")
    parser.add_argument("--data_dir", type=str, default="data/train", 
                        help="Path to data/train directory (default: data/train)")
    parser.add_argument("--ratio", type=float, default=0.2,
                        help="Proportion to move to validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    split_train_to_val(args.data_dir, val_ratio=args.ratio, seed=args.seed)