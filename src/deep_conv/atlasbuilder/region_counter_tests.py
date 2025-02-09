import unittest
import pandas as pd
import numpy as np
from deep_conv.atlasbuilder.find_marker_candidates import *

class TestRegionCounter(unittest.TestCase):
    def setUp(self):
        # Basic test data
        self.test_regions = pd.DataFrame({
            'startCpG': [100, 105, 110, 115],
            'endCpG':   [120, 125, 130, 135],
            'name': ['region1', 'region2', 'region3', 'region4'],
            'direction': ['U', 'U', 'U', 'U']
        })

    def test_basic_overlap(self):
        """Test basic pattern overlap with regions"""
        counter = RegionCounter(self.test_regions, min_cpgs=4)
        pattern = "CCCC"  # 4 CpGs
        overlaps = counter.find_overlapping_regions(110, pattern)
        self.assertEqual(len(overlaps), 3)  # Should overlap with regions 1,2,3
        
    def test_edge_overlaps(self):
        """Test patterns that overlap at region boundaries"""
        regions = pd.DataFrame({
            'startCpG': [100],
            'endCpG':   [110],
            'name': ['region1'],
            'direction': ['U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Pattern exactly at start
        overlaps = counter.find_overlapping_regions(100, "CCCC")
        self.assertEqual(len(overlaps), 1)
        
        # Pattern exactly ending at region end minus min_cpgs+1
        overlaps = counter.find_overlapping_regions(106, "CCCC")
        self.assertEqual(len(overlaps), 1)
        
        # Pattern starts before region but has enough CpGs in overlap
        overlaps = counter.find_overlapping_regions(98, "CCCCCC")
        self.assertEqual(len(overlaps), 1)
        
        # Pattern extends beyond region but doesn't have enough CpGs in overlap
        overlaps = counter.find_overlapping_regions(108, "CCCCCC")
        self.assertEqual(len(overlaps), 0)  # Not enough CpGs in overlap

        overlaps = counter.find_overlapping_regions(90, "CCCCCCCCCCCCCCCCCCCCCCCCCC")
        self.assertEqual(len(overlaps), 1)  

        overlaps = counter.find_overlapping_regions(95, "CCCCCC..........CCC")
        self.assertEqual(len(overlaps), 0)  

        overlaps = counter.find_overlapping_regions(95, "CCCCC.......CCCC")
        self.assertEqual(len(overlaps), 0)  

        overlaps = counter.find_overlapping_regions(95, "CCCCC.....CCCCC")
        self.assertEqual(len(overlaps), 1)  

    def test_multiple_overlaps(self):
        """Test pattern overlapping multiple regions"""
        regions = pd.DataFrame({
            'startCpG': [100, 105, 110],
            'endCpG':   [115, 120, 125],
            'name': ['r1', 'r2', 'r3'],
            'direction': ['U', 'U', 'U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Pattern overlapping first two regions
        overlaps = counter.find_overlapping_regions(102, "CCCCCCCCCC")
        self.assertEqual(len(overlaps), 2)  # Fixed: Third region doesn't have enough CpGs
        
        # Check exact overlaps
        lens = [o[2] for o in overlaps]  # Get overlap lengths
        self.assertTrue(all(l >= 4 for l in lens))  # All overlaps should have at least min_cpgs

    def test_boundary_conditions(self):
        """Test various boundary conditions"""
        regions = pd.DataFrame({
            'startCpG': [100, 110, 120],
            'endCpG':   [108, 118, 128],
            'name': ['r1', 'r2', 'r3'],
            'direction': ['U', 'U', 'U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Pattern exactly fitting remaining space
        overlaps = counter.find_overlapping_regions(104, "CCCC")
        self.assertEqual(len(overlaps), 1)
        
        # Pattern one CpG too long for remaining space
        overlaps = counter.find_overlapping_regions(105, "CCCC")
        self.assertEqual(len(overlaps), 0)
        
        # Pattern starting at last possible position
        overlaps = counter.find_overlapping_regions(124, "CCCC")
        self.assertEqual(len(overlaps), 1)
        
        # Pattern starting one position too late
        overlaps = counter.find_overlapping_regions(125, "CCCC")
        self.assertEqual(len(overlaps), 0)

    def test_min_cpgs_edge_cases(self):
        """Test edge cases around min_cpgs requirement"""
        counter = RegionCounter(self.test_regions, min_cpgs=4)
        
        # Pattern with exactly min_cpgs at region boundary
        overlaps = counter.find_overlapping_regions(116, "CCCC")
        self.assertEqual(len(overlaps), 4)
        
        # Pattern with more than min_cpgs but overlap has exactly min_cpgs
        overlaps = counter.find_overlapping_regions(116, "CCCCCC")
        self.assertEqual(len(overlaps), 4)
        
        # Pattern with more than min_cpgs but overlap has less than min_cpgs
        overlaps = counter.find_overlapping_regions(117, "CCCCCC")
        self.assertEqual(len(overlaps), 3)

    def test_pattern_types(self):
        """Test different pattern types and their overlaps"""
        counter = RegionCounter(self.test_regions, min_cpgs=4)
        
        # All methylated
        overlaps = counter.find_overlapping_regions(110, "CCCC")
        self.assertEqual(len(overlaps), 3)
        
        # All unmethylated
        overlaps = counter.find_overlapping_regions(110, "TTTT")
        self.assertEqual(len(overlaps), 3)
        
        # Mixed pattern
        overlaps = counter.find_overlapping_regions(110, "CTCT")
        self.assertEqual(len(overlaps), 3)
        
        # Invalid pattern
        overlaps = counter.find_overlapping_regions(110, "AAAA")
        self.assertEqual(len(overlaps), 0)

    def test_long_patterns(self):
        """Test behavior with long patterns"""
        counter = RegionCounter(self.test_regions, min_cpgs=4)
        
        # Long pattern overlapping multiple regions
        overlaps = counter.find_overlapping_regions(105, "CCCCCCCCCCCCCC")
        self.assertTrue(all(o[2] >= 4 for o in overlaps))  # All overlaps should have at least min_cpgs
        
        # Long pattern with partial overlaps
        overlaps = counter.find_overlapping_regions(95, "CCCCCCCCCCCCCCCCCC")
        self.assertTrue(all(o[2] >= 4 for o in overlaps))

    def test_patterns_with_dots(self):
        """Test handling of patterns containing dots ('.')"""
        regions = pd.DataFrame({
            'startCpG': [100, 105, 110],
            'endCpG':   [108, 113, 118],
            'name': ['r1', 'r2', 'r3'],
            'direction': ['U', 'U', 'U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Pattern with dots but enough valid CpGs
        overlaps = counter.find_overlapping_regions(102, "CC..CC")
        self.assertEqual(len(overlaps), 1)  # Should count 4 CpGs
        
        # Pattern with dots but not enough valid CpGs
        overlaps = counter.find_overlapping_regions(102, "C...C")
        self.assertEqual(len(overlaps), 0)  # Only 2 valid CpGs
        
        # Pattern with dots at overlap boundaries
        overlaps = counter.find_overlapping_regions(106, "..CCCC..")
        self.assertEqual(len(overlaps), 1)  
        
        # Long pattern with many dots
        overlaps = counter.find_overlapping_regions(104, "C.C.C.C.C")
        self.assertEqual(len(overlaps), 1)  # 5 valid CpGs despite dots

        # Pattern with dots in overlapping regions
        overlaps = counter.find_overlapping_regions(107, "CCCC...CCCC")
        # Should only count valid CpGs in each overlap, not dots
        for region, offset, length in overlaps:
            overlap_pat = "CCCC...CCCC"[offset:offset + length]
            valid_cpgs = sum(1 for c in overlap_pat if c in 'CT')
            self.assertGreaterEqual(valid_cpgs, 4)


    def test_pattern_counting_and_classification(self):
        """Test full pattern processing including UXM classification and counting"""
        regions = pd.DataFrame({
            'startCpG': [100, 105],
            'endCpG':   [110, 115],
            'name': ['r1', 'r2'],
            'direction': ['U', 'U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Unmethylated pattern
        counter.process_pattern("TTTT", 102, 3)  # count=3
        self.assertEqual(counter.counts[0]['u'], 3)
        self.assertEqual(counter.counts[0]['m'], 0)
        self.assertEqual(counter.counts[0]['x'], 0)
        
        # Methylated pattern
        counter.process_pattern("CCCC", 103, 2)  # count=2
        self.assertEqual(counter.counts[0]['u'], 3)
        self.assertEqual(counter.counts[0]['m'], 2)
        self.assertEqual(counter.counts[0]['x'], 0)
        
        # Intermediate pattern
        counter.process_pattern("CTCT", 104, 1)  # count=1
        self.assertEqual(counter.counts[0]['u'], 3)
        self.assertEqual(counter.counts[0]['m'], 2)
        self.assertEqual(counter.counts[0]['x'], 1)
        
        counter.process_pattern("CCCC", 107, 4)  # count=4
        self.assertEqual(counter.counts[0]['m'], 2) 
        self.assertEqual(counter.counts[1]['m'], 4) 

    def test_pattern_accumulation(self):
        """Test accumulation of pattern counts over multiple patterns"""
        regions = pd.DataFrame({
            'startCpG': [100],
            'endCpG':   [110],
            'name': ['r1'],
            'direction': ['U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        patterns = [
            ("TTTT", 102, 5),   # 5 unmethylated
            ("CCCC", 103, 3),   # 3 methylated
            ("CTCT", 104, 2),   # 2 intermediate
            ("TTTT", 102, 2),   # 2 more unmethylated (same position)
        ]
        
        for pattern, start, count in patterns:
            counter.process_pattern(pattern, start, count)
        
        self.assertEqual(counter.counts[0]['u'], 7)  # 5 + 2
        self.assertEqual(counter.counts[0]['m'], 3)
        self.assertEqual(counter.counts[0]['x'], 2)
        self.assertEqual(counter.patterns_counted, 4)


    def test_full_pattern_counting(self):
        """Test complete pattern counting including overlaps and min_cpgs requirement"""
        regions = pd.DataFrame({
            'startCpG': [100, 105],
            'endCpG':   [110, 115],
            'name': ['r1', 'r2'],
            'direction': ['U', 'U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Basic patterns - single region
        counter.process_pattern("TTTT", 102, 3)  # unmethylated, count=3
        self.assertEqual(counter.counts[0]['u'], 3)
        self.assertEqual(counter.counts[1]['u'], 0)
        
        counter.process_pattern("CCCC", 103, 2)  # methylated, count=2
        self.assertEqual(counter.counts[0]['m'], 2)
        self.assertEqual(counter.counts[1]['m'], 0)
        
        # Pattern that spans regions but only has enough CpGs for second region
        counter.process_pattern("CCCC", 107, 4)
        self.assertEqual(counter.counts[0]['m'], 2)  # unchanged
        self.assertEqual(counter.counts[1]['m'], 4)  # gets the count
        
        # Pattern that has enough CpGs for both regions
        counter.process_pattern("CCCCCC", 104, 1)
        self.assertEqual(counter.counts[0]['m'], 3)  # +1
        self.assertEqual(counter.counts[1]['m'], 5)  # +1


    def test_pattern_types_and_thresholds(self):
        """Test UXM classification based on methylation ratios"""
        regions = pd.DataFrame({
            'startCpG': [100],
            'endCpG':   [110],
            'name': ['r1'],
            'direction': ['U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Fully unmethylated (0%)
        counter.process_pattern("TTTT", 102, 1)
        self.assertEqual(counter.counts[0]['u'], 1)
        
        # Fully methylated (100%)
        counter.process_pattern("CCCC", 102, 1)
        self.assertEqual(counter.counts[0]['m'], 1)
        
        # Just below threshold for unmethylated
        counter.process_pattern("CTTT", 102, 1)
        self.assertEqual(counter.counts[0]['u'], 2)
        
        # Just above threshold for methylated
        counter.process_pattern("CCCCT", 102, 1)
        self.assertEqual(counter.counts[0]['m'], 2)
        
        # Intermediate
        counter.process_pattern("CCTT", 102, 1)
        self.assertEqual(counter.counts[0]['x'], 1)


    def test_boundary_patterns(self):
        """Test patterns at region boundaries with min_cpgs requirement"""
        regions = pd.DataFrame({
            'startCpG': [100, 110],
            'endCpG':   [108, 118],
            'name': ['r1', 'r2'],
            'direction': ['U', 'U']
        })
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Pattern starting at region boundary
        counter.process_pattern("CCCC", 100, 1)
        self.assertEqual(counter.counts[0]['m'], 1)
        
        # Pattern ending exactly at region boundary
        counter.process_pattern("CCCC", 104, 1)  # ends at 107
        self.assertEqual(counter.counts[0]['m'], 2)
        
        # Pattern crossing boundary but not enough CpGs in either region
        counter.process_pattern("CCCC", 106, 1)  # 106-109
        self.assertEqual(counter.counts[0]['m'], 2)  # unchanged
        self.assertEqual(counter.counts[1]['m'], 0)  # not counted


    def test_chunk_processing(self):
        """Test processing patterns in chunks and verifying final counts"""
        regions = pd.DataFrame({
            'startCpG': [100, 105],
            'endCpG':   [110, 115],
            'name': ['r1', 'r2'],
            'direction': ['U', 'U']
        })
        
        # Create test data similar to real chunk from pat file
        chunk1 = pd.DataFrame({
            'chr': ['chr1'] * 3,
            'start': [102, 103, 104],
            'pattern': ['TTTT', 'CCCC', 'CTCT'],
            'count': [3, 2, 1]
        })
        
        chunk2 = pd.DataFrame({
            'chr': ['chr1'] * 2,
            'start': [107, 108],
            'pattern': ['CCCC', 'TTTT'],
            'count': [4, 2]
        })
        
        # Process chunks
        counter = RegionCounter(regions, min_cpgs=4)
        
        # Process first chunk
        for _, row in chunk1.iterrows():
            counter.process_pattern(row['pattern'], row['start'], row['count'])
            
        # Verify intermediate state
        self.assertEqual(counter.counts[0]['u'], 3)  # first TTTT
        self.assertEqual(counter.counts[0]['m'], 2)  # CCCC
        self.assertEqual(counter.counts[0]['x'], 1)  # CTCT
        
        # Process second chunk
        for _, row in chunk2.iterrows():
            counter.process_pattern(row['pattern'], row['start'], row['count'])
        
        # Verify final state - pattern at 107 should affect both regions
        self.assertEqual(counter.counts[0]['u'], 3)
        self.assertEqual(counter.counts[0]['m'], 2)
        self.assertEqual(counter.counts[0]['x'], 1)
        self.assertEqual(counter.counts[1]['m'], 4)  # from pattern at 107
        self.assertEqual(counter.counts[1]['u'], 2)  # from pattern at 108

    def test_matrix_creation(self):
        """Test creation of UXM and coverage matrices"""
        regions = pd.DataFrame({
            'startCpG': [100, 105],
            'endCpG':   [110, 115],
            'name': ['r1', 'r2'],
            'direction': ['U', 'U']
        })
        
        # Define test patterns
        patterns = [
            ("TTTT", 102, 3),   # r1 only
            ("CCCC", 103, 2),   # r1 only
            ("CCCC", 106, 4),   # both regions
            ("TTTT", 108, 2)    # r2 only
        ]
        
        # Process patterns
        uxm_df, coverage_df, _ = process_patterns_for_test(regions, patterns, min_cpgs=4)
        
        # Verify results
        self.assertEqual(coverage_df.iloc[0]['value'], 9)  # 3 + 2 + 4
        self.assertEqual(coverage_df.iloc[1]['value'], 6)  # 4 + 2
        
        # Verify UXM proportions
        self.assertEqual(uxm_df.iloc[0]['value'], 3/9)  # u/(u+m) for r1

    def test_full_pipeline_with_filtering(self):
        """Test full pipeline including coverage filtering"""
        regions = pd.DataFrame({
            'startCpG': [100, 105, 110],
            'endCpG':   [110, 115, 120],
            'name': ['r1', 'r2', 'r3'],
            'direction': ['U', 'U', 'U']
        })
        
        patterns = [
            ("TTTT", 102, 3),   # r1 only
            ("CCCC", 106, 15),  # r1 and r2, high coverage
            ("TTTT", 111, 2),   # r2 and r3, low coverage
        ]
        
        # Process patterns
        _, coverage_df, _ = process_patterns_for_test(regions, patterns, min_cpgs=4)
        
        # Verify coverage
        self.assertEqual(coverage_df.iloc[0]['value'], 18) 
        self.assertEqual(coverage_df.iloc[1]['value'], 17) 
        self.assertEqual(coverage_df.iloc[2]['value'], 2)  


def process_patterns_for_test(regions_df: pd.DataFrame, patterns: List[Tuple[str, int, int]], min_cpgs: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Test version of process_pat_file that works with in-memory patterns"""
    counter = RegionCounter(regions_df, min_cpgs)
    
    # Process all patterns
    for pattern, start, count in patterns:
        counter.process_pattern(pattern, start, count)
    
    # Create results exactly as process_pat_file would
    results_uxm = []
    results_coverage = []
    for idx in range(len(regions_df)):
        counts = counter.counts[idx]
        total = sum(counts.values())
        if total > 0:
            results_uxm.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': counts['u'] / total,
                'cell_type': 'test'
            })
            results_coverage.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': total,
                'cell_type': 'test'
            })
        else:
            results_uxm.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': np.nan,
                'cell_type': 'test'
            })
            results_coverage.append({
                'name': regions_df.iloc[idx]['name'],
                'direction': regions_df.iloc[idx]['direction'],
                'value': 0,
                'cell_type': 'test'
            })            
    return pd.DataFrame(results_uxm), pd.DataFrame(results_coverage), 'test'

if __name__ == '__main__':
    unittest.main(verbosity=2)
def create_test_data():
    """Helper function to create test data"""
    # Create different test scenarios
    return {
        'basic': pd.DataFrame({
            'startCpG': [100, 110, 120],
            'endCpG':   [105, 115, 125],
            'name': ['r1', 'r2', 'r3'],
            'direction': ['U', 'U', 'U']
        }),
        'overlapping': pd.DataFrame({
            'startCpG': [100, 100, 100],
            'endCpG':   [110, 115, 120],
            'name': ['r1', 'r2', 'r3'],
            'direction': ['U', 'U', 'U']
        }),
        'edge_cases': pd.DataFrame({
            'startCpG': [100, 200, 300],
            'endCpG':   [150, 250, 350],
            'name': ['r1', 'r2', 'r3'],
            'direction': ['U', 'U', 'U']
        })
    }

if __name__ == '__main__':
    unittest.main(verbosity=2)