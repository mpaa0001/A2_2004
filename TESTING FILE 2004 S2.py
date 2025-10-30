

##QUESTION 2##
import time
class Analyser:
    def __init__(self, sequences):
        
        self.sequences = sequences
        
        N= len(sequences)
        max_length = 0
        for s in sequences:
            max_length = max(max_length, len(s))

        self.max_frequency = [0 for _ in range(max_length + 1)]
        self.best_pattern_location = [None for _ in range(max_length + 1)] 

        pattern_frequ_map = []

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
                        if not added_already:
                            song_indices.append(song_id)


                        frequency = len(song_indices)

                    if frequency > self.max_frequency[pattern_length]:
                        self.max_frequency[pattern_length] = frequency
                        self.best_pattern_location[pattern_length] = (song_id, start_index)
                    
                    
    def getFrequentPattern(self, K):
        if K >= len(self.best_pattern_location):
            return []
        if self.best_pattern_location[K] is None:
            return []
        

        location = self.best_pattern_location[K]
        song_index = location[0]
        start_index = location[1]   

        song_with_best_pattern = self.sequences[song_index]

        pattern_string = song_with_best_pattern[start_index:start_index + K]
        return list(pattern_string)
    
    

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

