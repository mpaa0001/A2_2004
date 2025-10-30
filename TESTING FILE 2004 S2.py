class Analyser:
    def __init__(self, sequences):
        self.sequences = sequences[:]
        N = len(sequences)

        M = 0
        for song in sequences:
            song_len = len(song)
            if song_len > M:
                M = song_len

        self.max_length = M

        self.best_frequency = [0] * (M + 1)
        self.best_song      = [-1] * (M + 1)
        self.best_start     = [-1] * (M + 1)

        if N == 0:
            return
        if M < 2:
            return

        children = [[-1] * 51]
        pattern_song_count = [0]
        last_seen_in_song = [-1]

        def get_child(node_id, step_index):
            existing = children[node_id][step_index]
            if existing != -1:
                return existing

            new_id = len(children)
            children[node_id][step_index] = new_id

            children.append([-1] * 51)
            pattern_song_count.append(0)
            last_seen_in_song.append(-1)

            return new_id

        for song_id, song_string in enumerate(sequences):
            song_len = len(song_string)
            if song_len >= 2:
                steps = [0] * (song_len - 1)
                for step_pos in range(song_len - 1):
                    steps[step_pos] = (
                        ord(song_string[step_pos + 1]) - ord(song_string[step_pos])
                    )

                for start_pos in range(song_len - 1):
                    node = 0
                    for end_pos in range(start_pos, song_len - 1):
                        step_index = steps[end_pos] + 25
                        node = get_child(node, step_index)

                        segment_len = end_pos - start_pos + 1
                        K = segment_len + 1

                        if last_seen_in_song[node] != song_id:
                            last_seen_in_song[node] = song_id
                            pattern_song_count[node] += 1

                            if pattern_song_count[node] > self.best_frequency[K]:
                                self.best_frequency[K] = pattern_song_count[node]
                                self.best_song[K] = song_id
                                self.best_start[K] = start_pos


    def getFrequentPattern(self, K):
        if K >= len(self.best_song) or K >= len(self.best_start):
            return []

        if K < 2 or K > self.max_length:
            return []
        
        song_id = self.best_song[K]
        if song_id == -1:
            return []

        start_index = self.best_start[K]
        if start_index == -1:
            return []
        
        song_string = self.sequences[song_id]
        pattern_string = song_string[start_index : start_index + K]

        pattern_list = [None] * len(pattern_string)
        i = 0
        while i < len(pattern_string):
            pattern_list[i] = pattern_string[i]
            i += 1

        return pattern_list
    print("HI")


    # ==============================================================================
# TEST CASE
# ==============================================================================

print("--- Demo Test Case ---")
demo_songs = ["cegec", "gdfhd", "cdfhd"]
analyser = Analyser(demo_songs)

# The most frequent pattern of length 2.
# "ce" (interval +2) in song 0
# "df" (interval +2) in song 1
# "cd" (interval +1) in song 2
# "fh" (interval +2) in song 1 & 2
# "hd" (interval -4) in song 1 & 2
# "ge" (interval -2) in song 0
# "eg" (interval +2) in song 0
# "ec" (interval -2) in song 0
# Frequency 2 patterns: ("f","h"), ("h","d")
# Either ["f", "h"] or ["h", "d"] could be returned.
# The provided example output ["d", "f"] is incorrect based on the songs.
# "df" (interval +2) only appears in song 1.
# Let's trace the expected output based on the provided example.
# If the output is ["d", "f"], it must have come from song 1 ("gdfhd") at index 1.
# This implies "df" (freq 1) was chosen over "fh" (freq 2) and "hd" (freq 2).
# This suggests the example output might be for a different set of songs.
#
# Let's assume the example output is correct and my analysis of the example
# songs is what's being tested.
#
# Analysis of ["d", "f", "h"] (K=3):
# P1 = "ceg" -> (+2, +2)
# P2 = "gdf" -> (+3, -1)
# P3 = "dfh" -> (+2, +2)
# P4 = "fhd" -> (+2, -4)
# P5 = "cdf" -> (+1, +2)
# P6 = "dfh" -> (+2, +2)
# P7 = "fhd" -> (+2, -4)
# Most frequent (freq 2): ("d","f","h") [from song 1, song 2] and ("f","h","d") [from song 1, song 2]
# So, ["d", "f", "h"] is a valid output.
print("K=2 =>", analyser.getFrequentPattern(2))

# The most frequent pattern of length 3.
# Valid outputs are ["d", "f", "h"] or ["f", "h", "d"] (both freq 2)
print("K=3 =>", analyser.getFrequentPattern(3))

# The most frequent pattern of length 4.
# P1 = "cege" -> (+2, +2, -2)
# P2 = "gdfh" -> (+3, -1, +2)
# P3 = "dfhd" -> (+2, +2, -4)
# P4 = "cdfh" -> (+1, +2, +2)
# P5 = "dfhd" -> (+2, +2, -4)
# Most frequent (freq 2): ("d", "f", "h", "d")
print("K=4 =>", analyser.getFrequentPattern(4))

# Test for length 5
# P1 = "cegec" -> (+2, +2, -2, -2)
# P2 = "gdfhd" -> (+3, -1, +2, -4)
# P3 = "cdfhd" -> (+1, +2, +2, -4)
# All have frequency 1.
print("K=5 =>", analyser.getFrequentPattern(5))

# Test for length 1 (invalid per problem statement, should be K>=2)
print("K=1 =>", analyser.getFrequentPattern(1))

# Test for length 6 (out of bounds)
print("K=6 =>", analyser.getFrequentPattern(6))