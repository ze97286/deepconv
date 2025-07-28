#!/usr/bin/env python3
"""
Performance test script to compare original vs optimized prepare_for_atlas functions.
"""

import time
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import the module
sys.path.append(str(Path(__file__).parent.parent))

from deep_conv.atlasbuilder.collect_markers_for_training import (
    prepare_for_atlas, 
    prepare_for_atlas_optimized, 
    prepare_for_atlas_ultra_optimized,
    prepare_for_atlas_fast
)

def run_performance_test(atlas_path, pat_dir, min_cpgs=4, prefix="test", threads=8):
    """
    Run performance comparison between original and optimized versions.
    """
    print("=" * 60)
    print("PERFORMANCE COMPARISON TEST")
    print("=" * 60)
    
    # Test 1: Original version
    print("\n1. Testing ORIGINAL version...")
    start_time = time.time()
    try:
        prepare_for_atlas(atlas_path, pat_dir, min_cpgs, f"{prefix}_original", threads)
        original_time = time.time() - start_time
        print(f"   Original version completed in {original_time:.2f} seconds")
    except Exception as e:
        print(f"   Original version failed: {e}")
        original_time = None
    
    # Test 2: Optimized version
    print("\n2. Testing OPTIMIZED version...")
    start_time = time.time()
    try:
        prepare_for_atlas_optimized(atlas_path, pat_dir, min_cpgs, f"{prefix}_optimized", threads)
        optimized_time = time.time() - start_time
        print(f"   Optimized version completed in {optimized_time:.2f} seconds")
    except Exception as e:
        print(f"   Optimized version failed: {e}")
        optimized_time = None
    
    # Test 3: Ultra-optimized version
    print("\n3. Testing ULTRA-OPTIMIZED version...")
    start_time = time.time()
    try:
        prepare_for_atlas_ultra_optimized(atlas_path, pat_dir, min_cpgs, f"{prefix}_ultra", threads)
        ultra_time = time.time() - start_time
        print(f"   Ultra-optimized version completed in {ultra_time:.2f} seconds")
    except Exception as e:
        print(f"   Ultra-optimized version failed: {e}")
        ultra_time = None
    
    # Test 4: Auto-detection version
    print("\n4. Testing AUTO-DETECTION version...")
    start_time = time.time()
    try:
        prepare_for_atlas_fast(atlas_path, pat_dir, min_cpgs, f"{prefix}_auto", threads)
        auto_time = time.time() - start_time
        print(f"   Auto-detection version completed in {auto_time:.2f} seconds")
    except Exception as e:
        print(f"   Auto-detection version failed: {e}")
        auto_time = None
    
    # Print results
    print("\n" + "=" * 60)
    print("PERFORMANCE RESULTS")
    print("=" * 60)
    
    if original_time and optimized_time:
        speedup = original_time / optimized_time
        print(f"Optimized vs Original: {speedup:.1f}x speedup")
    
    if original_time and ultra_time:
        speedup = original_time / ultra_time
        print(f"Ultra-optimized vs Original: {speedup:.1f}x speedup")
    
    if original_time and auto_time:
        speedup = original_time / auto_time
        print(f"Auto-detection vs Original: {speedup:.1f}x speedup")
    
    print("\nDetailed timing:")
    if original_time:
        print(f"  Original: {original_time:.2f}s")
    if optimized_time:
        print(f"  Optimized: {optimized_time:.2f}s")
    if ultra_time:
        print(f"  Ultra-optimized: {ultra_time:.2f}s")
    if auto_time:
        print(f"  Auto-detection: {auto_time:.2f}s")

def main():
    """Main function to run performance test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance test for prepare_for_atlas functions")
    parser.add_argument("--atlas_path", type=str, required=True, help="Path to atlas file")
    parser.add_argument("--pat_dir", type=str, required=True, help="Directory containing pat files")
    parser.add_argument("--min_cpgs", type=int, default=4, help="Minimum CpGs required")
    parser.add_argument("--prefix", type=str, default="perf_test", help="Output file prefix")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads to use")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.atlas_path):
        print(f"Error: Atlas file not found: {args.atlas_path}")
        return
    
    if not os.path.exists(args.pat_dir):
        print(f"Error: Pat directory not found: {args.pat_dir}")
        return
    
    # Run performance test
    run_performance_test(
        args.atlas_path, 
        args.pat_dir, 
        args.min_cpgs, 
        args.prefix, 
        args.threads
    )

if __name__ == "__main__":
    main() 