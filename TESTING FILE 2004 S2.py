
"""
This file contains the Analyser class for finding frequent music patterns.

WARNING: This version has been modified to strictly AVOID dictionaries and sets
as per the assignment's general rules. This means it implements a compliant
solution, but it CANNOT meet the O(NM^2) time complexity for __init__.
The time complexity of __init__ is now approximately O(N^2 * M^4) in the
worst case, because finding and updating entries in a list-based map
is O(P) and O(N) respectively (where P is num_patterns, up to O(NM^2)).

This solution prioritizes the "no dict/set" rule over the time complexity.
"""
import time # Added for the test harness

class Analyser:
    """
    Analyser class to find the most frequent melodic patterns.
    (No-dict/set version)

    Attributes:
        sequences (list[str]): The original list of song sequences.
        max_frequency (list[int]): Stores the highest frequency found for each length K.
        best_pattern_location (list):
            Maps length K (as an index) to a tuple (s_idx, i) representing
            the location of the most frequent pattern.
    """

    def __init__(self, sequences):
        """
        Constructor for Analyser.

        Time Complexity:
            O(N^2 * M^4) in a bad case. The N*M^2 loops have an
            inner loop to search 'pattern_frequ_map' (O(P_keys) ~ O(NM^2))
            and another to check for song_id uniqueness (O(N)).
        
        Space Complexity (during __init__):
            O(N * M^2) for the temporary 'pattern_frequ_map' list.
        
        Space Complexity (Final Object):
            O(N*M), bounded by storing:
            1. self.sequences: O(N*M)
            2. self.max_frequency: O(M)
            3. self.best_pattern_location: O(M)
            Total: O(NM + M) = O(NM)
        """
        self.sequences = sequences
        
        N = len(sequences)
        max_length = 0
        # This loop is *only* to find max_length
        for s in sequences:
            max_length = max(max_length, len(s))

        # --- REPLACED dict WITH list ---
        self.max_frequency = [0 for _ in range(max_length + 1)]
        self.best_pattern_location = [None for _ in range(max_length + 1)] 

        # --- REPLACED dict WITH list ---
        # This will store entries: [ (k, hash), [song_indices_list] ]
        pattern_frequ_map = [] 

        # A prime base for the rolling hash (using your choice of 37)
        BASE = 37
        
        for s_idx, song in enumerate(self.sequences):
            n = len(song)
            for i in range(n): # O(M) - start of subsequence
                current_hash = 0
                for j in range(i + 1, n): # O(M) - end of subsequence
                    diff = ord(song[j]) - ord(song[j - 1])
                    current_hash = (current_hash * BASE) + diff
                    k = j - i + 1
                    key = (k, current_hash)

                    # --- REPLACED dict lookup WITH list search (Slow) ---
                    found_entry = None
                    for entry in pattern_frequ_map:
                        if entry[0] == key:
                            found_entry = entry
                            break
                    
                    if found_entry is None:
                        found_entry = [key, []]
                        pattern_frequ_map.append(found_entry)
                    
                    song_list = found_entry[1]

                    # --- REPLACED set.add() WITH list search (Slow) ---
                    is_present = False
                    for song_id in song_list:
                        if song_id == s_idx:
                            is_present = True
                            break
                    
                    if not is_present:
                        song_list.append(s_idx)
                    # --- End of replacement ---
                    
                    freq = len(song_list)
                    
                    # --- FIXED LOGICAL BUG ---
                    # This check is now outside the if/elif/else,
                    # so it runs every time.
                    if freq > self.max_frequency[k]:
                        self.max_frequency[k] = freq
                        # Store in the list using k as the index
                        self.best_pattern_location[k] = (s_idx, i)

    def getFrequentPattern(self, K):
        """
        Returns the most frequent pattern of a specific length K.

        Time Complexity:
            O(K). We perform a list lookup O(1) avg,
            then slice a string of length K, O(K).
        
        Aux Space Complexity:
            O(K) for the returned list.
        """
        
        # O(1) time list lookup
        if K >= len(self.best_pattern_location) or self.best_pattern_location[K] is None:
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
    
    print("--- Analyser Test (No Dict/Set Version) ---")
    
    demo_songs = ["cegec", "gdfhd", "cdfhd"]
    
    print(f"Songs: {demo_songs}")
    try:
        start_time = time.time()
        analyser = Analyser(demo_songs)
        end_time = time.time()
        print(f"__init__ took: {end_time - start_time:.6f} seconds")

        # Test K=2
        pattern_2 = analyser.getFrequentPattern(2)
        print(f"K=2 => {pattern_2}")
        if "".join(pattern_2) in ("df", "fh", "ce", "eg"):
             print("K=2 Test: PASS (Found a valid pattern)")
        else:
             print(f"K=2 Test: UNEXPECTED (Got {''.join(pattern_2)})")

        # Test K=3
        pattern_3 = analyser.getFrequentPattern(3)
        print(f"K=3 => {pattern_3}")
        if "".join(pattern_3) in ("dfh", "ceg"):
             print("K=3 Test: PASS (Found a valid pattern)")
        else:
             print(f"K-3 Test: UNEXPECTED (Got {''.join(pattern_3)})")

        # Test K=4
        pattern_4 = analyser.getFrequentPattern(4)
        print(f"K=4 => {pattern_4}")
        if "".join(pattern_4) == "dfhd":
             print("K-4 Test: PASS (Found the only valid pattern)")
        else:
             print(f"K-4 Test: UNEXPECTED (Got {''.join(pattern_4)})")
             
        # Test K=5
        pattern_5 = analyser.getFrequentPattern(5)
        print(f"K=5 => {pattern_5}")
        if "".join(pattern_5) in ("cegec", "gdfhd", "cdfhd"):
             print("K-5 Test: PASS")
        else:
             print(f"K-5 Test: UNEXPECTED (Got {''.join(pattern_5)})")
             
        # Test K=6 (Out of bounds)
        pattern_6 = analyser.getFrequentPattern(6)
        print(f"K-6 => {pattern_6}")
        if pattern_6 == []:
             print("K-6 Test: PASS (Correctly returned empty list)")
        else:
             print(f"K-6 Test: FAIL (Got {pattern_6})")

    except Exception as e:
        print(f"\n--- !! An error occurred: {e} !! ---")



"""
This file contains the Analyser class for finding frequent music patterns.

WARNING: This version has been modified to strictly AVOID dictionaries and sets
as per the assignment's general rules. This means it implements a compliant
solution, but it CANNOT meet the O(NM^2) time complexity for __init__.
The time complexity of __init__ is now approximately O(N^2 * M^4) in the
worst case, because finding and updating entries in a list-based map
is O(P) and O(N) respectively (where P is num_patterns, up to O(NM^2)).

This solution prioritizes the "no dict/set" rule over the time complexity.
"""
import time # Added for the test harness

class Analyser:
    """
    Analyser class to find the most frequent melodic patterns.
    (No-dict/set version)

    Attributes:
        sequences (list[str]): The original list of song sequences.
        max_frequency (list[int]): Stores the highest frequency found for each length K.
        best_pattern_location (list):
            Maps length K (as an index) to a tuple (s_idx, i) representing
            the location of the most frequent pattern.
    """

    def __init__(self, sequences):
        """
        Constructor for Analyser.

        Time Complexity:
            O(N^2 * M^4) in a bad case. The N*M^2 loops have an
            inner loop to search 'pattern_frequ_map' (O(P_keys) ~ O(NM^2))
            and another to check for song_id uniqueness (O(N)).
        
        Space Complexity (during __init__):
            O(N * M^2) for the temporary 'pattern_frequ_map' list.
        
        Space Complexity (Final Object):
            O(N*M), bounded by storing:
            1. self.sequences: O(N*M)
            2. self.max_frequency: O(M)
            3. self.best_pattern_location: O(M)
            Total: O(NM + M) = O(NM)
        """
        self.sequences = sequences
        
        N = len(sequences)
        max_length = 0
        # This loop is *only* to find max_length
        for s in sequences:
            max_length = max(max_length, len(s))

        # --- REPLACED dict WITH list ---
        self.max_frequency = [0 for _ in range(max_length + 1)]
        self.best_pattern_location = [None for _ in range(max_length + 1)] 

        # --- REPLACED dict WITH list ---
        # This will store entries: [ (k, hash), [song_indices_list] ]
        pattern_frequ_map = [] 

        # A prime base for the rolling hash (using your choice of 37)
        BASE = 37
        
        for song_id, song in enumerate(self.sequences):
            song_len = len(song)

            for start_index in range(song_len): # O(M) - start of subsequence
                rolling_hash = 0

                for end_index in range(start_index + 1, song_len): # O(M) - end of subsequence
                    interval = ord(song[end_index]) - ord(song[end_index - 1])
                    rolling_hash = (rolling_hash * BASE) + interval

                    pattern_length = end_index - start_index + 1
                    pattern_key = (pattern_length, rolling_hash)

                    # --- REPLACED dict lookup WITH list search (Slow) ---
                    entry_found = None
                    for entry in pattern_frequ_map:
                        if entry[0] == pattern_key:
                            entry_found = entry
                            break

                    if entry_found is None:
                        song_indices = [song_id]
                        pattern_frequ_map.append([pattern_key, song_indices])
                        frequency = 1
                    else:
                        song_indices = entry_found[1]
                        added_already = False

                        for previous_id in song_indices:
                            if previous_id == song_id:
                                added_already = True
                                break
                        
                        # --- BUG FIX (Start) ---
                        # Was missing this section
                        if not added_already:
                            song_indices.append(song_id)

                        # 'frequency' must be calculated *after* potentially adding
                        frequency = len(song_indices)
                        # --- BUG FIX (End) ---

                    if frequency > self.max_frequency[pattern_length]:
                        self.max_frequency[pattern_length] = frequency
                        self.best_pattern_location[pattern_length] = (song_id, start_index)

    def getFrequentPattern(self, K):
        """
        Returns the most frequent pattern of a specific length K.

        Time Complexity:
            O(K). We perform a list lookup O(1) avg,
            then slice a string of length K, O(K).
        
        Aux Space Complexity:
            O(K) for the returned list.
        """
        
        # O(1) time list lookup
        if K >= len(self.best_pattern_location) or self.best_pattern_location[K] is None:
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
    
    print("--- Analyser Test (No Dict/Set Version) ---")
    
    demo_songs = ["cegec", "gdfhd", "cdfhd"]
    
    print(f"Songs: {demo_songs}")
    try:
        start_time = time.time()
        analyser = Analyser(demo_songs)
        end_time = time.time()
        print(f"__init__ took: {end_time - start_time:.6f} seconds")

        # Test K=2
        pattern_2 = analyser.getFrequentPattern(2)
        print(f"K=2 => {pattern_2}")
        if "".join(pattern_2) in ("df", "fh", "ce", "eg"):
             print("K=2 Test: PASS (Found a valid pattern)")
        else:
             print(f"K=2 Test: UNEXPECTED (Got {''.join(pattern_2)})")

        # Test K=3
        pattern_3 = analyser.getFrequentPattern(3)
        print(f"K=3 => {pattern_3}")
        if "".join(pattern_3) in ("dfh", "ceg"):
             print("K-3 Test: PASS (Found a valid pattern)")
        else:
             print(f"K-3 Test: UNEXPECTED (Got {''.join(pattern_3)})")

        # Test K=4
        pattern_4 = analyser.getFrequentPattern(4)
        print(f"K=4 => {pattern_4}")
        if "".join(pattern_4) == "dfhd":
             print("K-4 Test: PASS (Found the only valid pattern)")
        else:
             print(f"K-4 Test: UNEXPECTED (Got {''.join(pattern_4)})")
             
        # Test K=5
        pattern_5 = analyser.getFrequentPattern(5)
        print(f"K=5 => {pattern_5}")
        if "".join(pattern_5) in ("cegec", "gdfhd", "cdfhd"):
             print("K-5 Test: PASS")
        else:
             print(f"K-5 Test: UNEXPECTED (Got {''.join(pattern_5)})")
             
        # Test K=6 (Out of bounds)
        pattern_6 = analyser.getFrequentPattern(6)
        print(f"K-6 => {pattern_6}")
        if pattern_6 == []:
             print("K-6 Test: PASS (Correctly returned empty list)")
        else:
             print(f"K-6 Test: FAIL (Got {pattern_6})")

    except Exception as e:
        print(f"\n--- !! An error occurred: {e} !! ---")
