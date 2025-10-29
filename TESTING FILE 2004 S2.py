
class Analyser:
    

    def __init__(self, sequences):
        
        self.sequences = sequences
        
        N = len(sequences)
        M_max = 0
        if N > 0:
            for s in sequences:
                if len(s) > M_max:
                    M_max = len(s)
        
        # O(M) space - for final object
        self.max_freq = [0] * (M_max + 1)
        # O(M) space - for final object
        self.best_pattern_location = {} # K -> (song_index, start_index)

        # O(NM^2) temporary space
        pattern_counts = {} # (k, hash) -> set(song_indices)

        # A prime base for the rolling hash
        BASE = 31 
        
        for s_idx, song in enumerate(self.sequences):
            n = len(song)
            for i in range(n): # O(M) - start of subsequence
                current_hash = 0
                for j in range(i + 1, n): # O(M) - end of subsequence
                    # This is an O(1) operation
                    # Calculate the difference from the *previous* note
                    diff = ord(song[j]) - ord(song[j - 1])
                    
                    # Update rolling hash for the *difference sequence*
                    # H(d1, d2) = d1*B + d2
                    # H(d1, d2, d3) = (d1*B + d2)*B + d3 = H(d1,d2)*B + d3
                    current_hash = (current_hash * BASE) + diff

                    k = j - i + 1 # Length of the note pattern (k >= 2)
                    key = (k, current_hash)

                    if key not in pattern_counts:
                        pattern_counts[key] = set()
                    
                    pattern_counts[key].add(s_idx)
                    
                    freq = len(pattern_counts[key])
                    
                    # Check if this is the new best for length K
                    if freq > self.max_freq[k]:
                        self.max_freq[k] = freq
                        # Store the *location* of this pattern, not the
                        # pattern itself, to save space.
                        self.best_pattern_location[k] = (s_idx, i)

    def getFrequentPattern(self, K):
        
        
        # O(1) average time lookup
        if K not in self.best_pattern_location:
            return [] # No pattern of length K found

        # O(1) lookup
        s_idx, i = self.best_pattern_location[K]
        
        # O(1) lookup
        best_song = self.sequences[s_idx]
        
        # O(K) time to slice the string
        pattern_str = best_song[i : i + K]
        
        # O(K) time to convert string to list
        return list(pattern_str)

# =============================================================================
# --- TEST HARNESS ---
# You can run this file to test the Analyser class.
# =============================================================================

if __name__ == "__main__":
    
    print("--- Analyser Test ---")
    
    demo_songs = ["cegec", "gdfhd", "cdfhd"]
    
    print(f"Songs: {demo_songs}")
    try:
        analyser = Analyser(demo_songs)

        # Test K=2
        pattern_2 = analyser.getFrequentPattern(2)
        print(f"K=2 => {pattern_2}")
        # Expected: ['d', 'f'] or ['f', 'h'] or ['c', 'e'] etc.
        # "df" (2), "fh" (2), "ce" (1), "eg" (1), "ge" (1), "ec" (1)
        # "gd" (-3), "df" (2), "fh" (2), "hd" (-4)
        # "cd" (1), "df" (2), "fh" (2), "hd" (-4)
        # Most frequent diffs: (2) with freq 3.
        # Possible patterns: ['d', 'f'], ['f', 'h'], ['c', 'e'], ['e', 'g']
        if "".join(pattern_2) in ("df", "fh", "ce", "eg"):
             print("K=2 Test: PASS (Found a valid pattern)")
        else:
             print(f"K=2 Test: UNEXPECTED (Got {''.join(pattern_2)})")


        # Test K=3
        pattern_3 = analyser.getFrequentPattern(3)
        print(f"K=3 => {pattern_3}")
        # Expected: ['d', 'f', 'h'] or ['c', 'e', 'g']
        # "ceg" (2,2) - song 0
        # "dfh" (2,2) - song 1, 2
        # Freq of (2,2) is 3.
        if "".join(pattern_3) in ("dfh", "ceg"):
             print("K=3 Test: PASS (Found a valid pattern)")
        else:
             print(f"K=3 Test: UNEXPECTED (Got {''.join(pattern_3)})")

        # Test K=4
        pattern_4 = analyser.getFrequentPattern(4)
        print(f"K=4 => {pattern_4}")
        # Expected: ['d', 'f', 'h', 'd']
        # "cege" (2, -2, 2) - song 0
        # "dfhd" (2, 2, -4) - song 1, 2
        # Freq of (2,2,-4) is 2.
        if "".join(pattern_4) == "dfhd":
             print("K=4 Test: PASS (Found the only valid pattern)")
        else:
             print(f"K=4 Test: UNEXPECTED (Got {''.join(pattern_4)})")
             
        # Test K=5
        pattern_5 = analyser.getFrequentPattern(5)
        print(f"K=5 => {pattern_5}")
        if "".join(pattern_5) in ("cegec", "gdfhd", "cdfhd"):
             print("K=5 Test: PASS")
        else:
             print(f"K=5 Test: UNEXPECTED (Got {''.join(pattern_5)})")
             
        # Test K=6 (Out of bounds)
        pattern_6 = analyser.getFrequentPattern(6)
        print(f"K=6 => {pattern_6}")
        if pattern_6 == []:
             print("K=6 Test: PASS (Correctly returned empty list)")
        else:
             print(f"K=6 Test: FAIL (Got {pattern_6})")

    except Exception as e:
        print(f"\n--- !! An error occurred: {e} !! ---")
