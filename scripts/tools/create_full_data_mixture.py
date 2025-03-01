#!/usr/bin/env python3
"""
create_complete_mixture.py - A utility to create a dataset mixture that includes ALL files 
from ALL source datasets.

Unlike the standard create_data_mixture.py which tries to achieve specific weights and stops
when a dataset is exhausted, this script ensures that every single file from every dataset
is included in the final mixture.

Example usage:
    python3 create_complete_mixture.py \
        --folders \
            /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/finemath-3plus-merge \
            /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/starcoder-extras-merge \
            /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/gutenberg \
        --output datasets/mixtures/my_complete_mixture \
        --verbose
"""

import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple


def get_bin_files(folder_path: str) -> List[str]:
    """
    Recursively find all .bin files in the specified folder and its subfolders.
    
    Args:
        folder_path: Path to the folder to search
        
    Returns:
        List of absolute paths to all .bin files
    """
    bin_files = []
    # Walk through the directory tree
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Only include .bin files
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    
    # Sort for consistent results
    bin_files.sort()
    return bin_files


def calculate_folder_size(folder_path: str) -> Tuple[int, int]:
    """
    Calculate the total size of .bin files in a folder and count files.
    
    Args:
        folder_path: Path to the folder
        
    Returns:
        Tuple of (total size in bytes, number of files)
    """
    bin_files = get_bin_files(folder_path)
    total_size = sum(os.path.getsize(file) for file in bin_files)
    return total_size, len(bin_files)


def human_readable_size(size_in_bytes: int) -> str:
    """
    Convert bytes to a human-readable string with appropriate units.
    
    Args:
        size_in_bytes: Size in bytes
        
    Returns:
        String with size and unit (e.g., "1.23 GB")
    """
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    unit_index = 0
    
    while size_in_bytes >= 1024 and unit_index < len(units) - 1:
        size_in_bytes /= 1024
        unit_index += 1
    
    return f"{size_in_bytes:.2f} {units[unit_index]}"


def create_symlink(source: str, target: str) -> bool:
    """
    Create a symbolic link if the target doesn't exist.
    
    Args:
        source: Path to the source file
        target: Path where the symlink should be created
        
    Returns:
        True if the symlink was created, False otherwise
    """
    if os.path.exists(target):
        return False
    
    # Ensure the target directory exists
    Path(os.path.dirname(target)).mkdir(parents=True, exist_ok=True)
    
    try:
        os.symlink(source, target)
        return True
    except Exception as e:
        print(f"Error creating symlink from {source} to {target}: {str(e)}")
        return False


def create_complete_mixture(
    folders: List[str], 
    output_folder: str, 
    verbose: bool = False
) -> Dict:
    """
    Create a mixture that includes ALL files from ALL datasets.
    
    Args:
        folders: List of folder paths containing dataset files
        output_folder: Path where the mixture will be created
        verbose: If True, print detailed progress information
        
    Returns:
        Dictionary with statistics about the created mixture
    """
    # Create the output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Initialize statistics
    stats = {
        "total_files": 0,
        "total_size_bytes": 0,
        "datasets": {},
        "start_time": time.time()
    }
    
    if verbose:
        print(f"Creating complete mixture in: {output_folder}")
        print(f"Processing {len(folders)} source folders:")
    
    # Process each input folder
    for folder in folders:
        # Get the base name to use as dataset name in the mixture
        dataset_name = os.path.basename(folder)
        output_dataset_dir = os.path.join(output_folder, dataset_name)
        
        if verbose:
            print(f"\nProcessing dataset: {dataset_name} from {folder}")
        
        # Find all .bin files in this folder
        bin_files = get_bin_files(folder)
        
        if not bin_files:
            if verbose:
                print(f"  Warning: No .bin files found in {folder}")
            continue
        
        # Initialize dataset stats
        dataset_stats = {
            "source_folder": folder,
            "files_count": len(bin_files),
            "dataset_size_bytes": 0,
            "symlinks_created": 0
        }
        
        if verbose:
            print(f"  Found {len(bin_files)} .bin files")
        
        # Create symlinks for each file
        for bin_file in bin_files:
            # Get the relative path within the source folder
            relative_path = os.path.relpath(bin_file, folder)
            symlink_path = os.path.join(output_dataset_dir, relative_path)
            
            # Create symlink for .bin file
            bin_size = os.path.getsize(bin_file)
            if create_symlink(bin_file, symlink_path):
                dataset_stats["symlinks_created"] += 1
                dataset_stats["dataset_size_bytes"] += bin_size
                stats["total_size_bytes"] += bin_size
                stats["total_files"] += 1
            
            # Create symlink for corresponding .idx file if it exists
            idx_file = bin_file.replace('.bin', '.idx')
            idx_symlink = symlink_path.replace('.bin', '.idx')
            if os.path.exists(idx_file):
                create_symlink(idx_file, idx_symlink)
        
        # Save dataset stats
        stats["datasets"][dataset_name] = dataset_stats
        
        if verbose:
            print(f"  Created {dataset_stats['symlinks_created']} symlinks")
            print(f"  Dataset size: {human_readable_size(dataset_stats['dataset_size_bytes'])}")
    
    # Calculate elapsed time
    stats["elapsed_time"] = time.time() - stats["start_time"]
    
    # Generate a summary
    if verbose:
        print("\nMixture Creation Summary:")
        print(f"  Total datasets: {len(stats['datasets'])}")
        print(f"  Total files: {stats['total_files']}")
        print(f"  Total size: {human_readable_size(stats['total_size_bytes'])}")
        print(f"  Estimated tokens: {stats['total_size_bytes'] // 4:,} (assuming 4 bytes per token)")
        print(f"  Creation time: {stats['elapsed_time']:.2f} seconds")
    
    # Create a summary file in the output directory
    summary_path = os.path.join(output_folder, "mixture_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Complete Dataset Mixture Summary\n")
        f.write(f"================================\n")
        f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total datasets: {len(stats['datasets'])}\n")
        f.write(f"Total files: {stats['total_files']}\n")
        f.write(f"Total size: {human_readable_size(stats['total_size_bytes'])}\n")
        f.write(f"Estimated tokens: {stats['total_size_bytes'] // 4:,} (assuming 4 bytes per token)\n\n")
        
        f.write("Dataset Details:\n")
        for dataset, dataset_stats in stats["datasets"].items():
            f.write(f"- {dataset}:\n")
            f.write(f"  Source: {dataset_stats['source_folder']}\n")
            f.write(f"  Files: {dataset_stats['files_count']}\n")
            f.write(f"  Size: {human_readable_size(dataset_stats['dataset_size_bytes'])}\n")
            f.write(f"  Percentage of total: {(dataset_stats['dataset_size_bytes'] / stats['total_size_bytes'] * 100):.2f}%\n\n")
    
    if verbose:
        print(f"\nMixture summary written to: {summary_path}")
    
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a complete mixture with ALL files from ALL datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python3 create_complete_mixture.py \\
      --folders \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/finemath-3plus-merge \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/starcoder-extras-merge \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/starcoder-threshold-0-merge \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-edu-score-2-filterrobots-merge \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/euro-high \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/euro-mid \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/other-high \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/swissai-fineweb-2-quality_33-filterrobots-merge/rest \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/poison \\
          /capstor/store/cscs/swissai/a06/datasets_tokenized/megatron/sai/gutenberg \\
      --output datasets/mixtures/my_complete_mixture \\
      --verbose

This will create a mixture containing ALL files from ALL the specified datasets.
Unlike the standard mixture creation script, this ensures no dataset is exhausted early.
The actual proportions during training will reflect the natural size proportions of your datasets.
        """
    )
    parser.add_argument(
        "--folders", 
        nargs='+', 
        required=True, 
        help="List of folders containing dataset files"
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Output folder where the mixture will be created"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    create_complete_mixture(args.folders, args.output, args.verbose)