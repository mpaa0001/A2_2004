class Analyser:
    def __init__(self, sequences):
        """
        Preprocesses all note sequences.

        Complexity goals:
        - Time: O(N * M^2)
        - Peak space during preprocessing: O(N * M^2)
        - Final stored space after init: O(N * M)

        Where:
        - N = number of songs
        - M = max song length
        """

        # Store original sequences so we can reconstruct patterns later.
        # This is O(N*M) space.
        self.sequences = sequences[:]  # shallow copy
        N = len(sequences)

        # Handle edge case: no songs
        if N == 0:
            self.max_len = 0
            self.best_freq = []
            self.best_song = []
            self.best_start = []
            return

        # Find M = max length of any sequence
        M = 0
        i = 0
        while i < N:
            length_i = len(sequences[i])
            if length_i > M:
                M = length_i
            i += 1
        self.max_len = M

        # We will compute, for each K (pattern length in NOTES, K >= 2):
        # - best_freq[K]: highest number of DISTINCT songs that contain
        #                 some transposition-equivalent pattern of length K
        # - best_song[K]: which song first achieved that best
        # - best_start[K]: where in that song it starts
        #
        # These are size M+1 so we can index directly by K.
        self.best_freq = [0] * (M + 1)
        self.best_song = [-1] * (M + 1)
        self.best_start = [-1] * (M + 1)

        # -------------------------------
        # BUILD TRIE OF INTERVAL PATTERNS
        # -------------------------------
        #
        # Each node in the trie corresponds to a sequence of intervals
        # (like [+2, +2, -1, ...]).
        #
        # Node storage (parallel lists, index = node_id):
        #   children[node_id] = list of length 51
        #       children[node_id][d] = next node index, or -1 if none.
        #       Here d is interval_code in [0..50] representing delta [-25..25].
        #
        #   freq[node_id] = in how many DISTINCT songs this pattern appears
        #
        #   last_song_seen[node_id] = last song_id that updated freq for this node
        #       (to avoid counting the same song multiple times)
        #
        # Root is node 0 and represents "empty interval sequence".

        children = [[-1] * 51]   # node 0
        freq = [0]
        last_song_seen = [-1]

        def get_child(node_id, interval_code):
            """
            Follow child with this interval_code. Create it if needed.

            interval_code is (delta + 25), so it is in [0..50].

            Returns: child_node_id (int)
            """
            next_id = children[node_id][interval_code]
            if next_id == -1:
                # Create new node
                next_id = len(children)
                children[node_id][interval_code] = next_id
                children.append([-1] * 51)
                freq.append(0)
                last_song_seen.append(-1)
            return next_id

        # ---------------------------------
        # MAIN PREPROCESSING LOOP
        # ---------------------------------
        #
        # For each song:
        #   1. Compute its interval array: diffs[j] = seq[j+1]-seq[j]
        #   2. For every start index in diffs:
        #        Walk forward, extending substring one interval at a time.
        #        Update trie nodes along this walk.
        #
        # If we're at substring diffs[start : end+1] (length L = end-start+1),
        # that corresponds to a NOTE pattern of length K = L+1.
        #
        # Every time we extend, we:
        #   - Mark that this trie node appears in this song (distinct count)
        #   - If its distinct-count freq becomes the best for this K,
        #     record (song_id, start_index) for reconstruction.

        song_id = 0
        while song_id < N:
            song = sequences[song_id]
            m = len(song)

            # If the song is shorter than 2 notes, it can't form a K>=2 pattern.
            if m >= 2:
                # Build interval array for this song:
                # diffs[j] = ord(song[j+1]) - ord(song[j]), range [-25..25]
                diffs = [0] * (m - 1)
                j = 0
                while j < m - 1:
                    diffs[j] = (ord(song[j + 1]) - ord(song[j]))
                    j += 1

                # Enumerate all substrings of diffs.
                # For each start in [0 .. m-2], we walk down the trie
                # as we extend end from start .. m-2.
                start = 0
                while start < (m - 1):
                    node = 0  # start at root of trie for each new start
                    end = start
                    while end < (m - 1):
                        # Convert interval -25..25 to code 0..50
                        interval_code = diffs[end] + 25
                        # step/create child
                        node = get_child(node, interval_code)

                        # Length of this interval-substring
                        L = end - start + 1
                        # Corresponding NOTE pattern length is K = L + 1
                        K = L + 1

                        # Count distinct songs for this node
                        if last_song_seen[node] != song_id:
                            last_song_seen[node] = song_id
                            freq[node] += 1

                            # If this subsequence is now the best for length K,
                            # remember its location.
                            if freq[node] > self.best_freq[K]:
                                self.best_freq[K] = freq[node]
                                self.best_song[K] = song_id
                                # IMPORTANT:
                                # start is an index in diffs. The pattern's first
                                # note index in the original song is also `start`.
                                self.best_start[K] = start

                        end += 1
                    start += 1

            song_id += 1

        # Done preprocessing.
        # We intentionally DO NOT store the trie (children/freq/etc.) on self.
        # That big structure was only needed to compute the best patterns.
        # After __init__ finishes, it gets garbage collected.
        #
        # Final object space:
        #   - self.sequences: O(N*M)
        #   - self.best_* arrays: O(M)
        # Total: O(N*M).


    def getFrequentPattern(self, K):
        """
        Return the most frequent transposition-equivalent pattern of length K
        as a list of characters. If multiple are tied, we return one of them.

        Worst-case time: O(K).
        """

        # Guard 1: K must be at least 2 (problem spec) and at most max_len.
        if K < 2 or K > self.max_len:
            return []

        # Guard 2: make sure our arrays are long enough (defensive in case of indentation issues).
        if K >= len(self.best_song) or K >= len(self.best_start):
            return []

        song_id = self.best_song[K]
        start_idx = self.best_start[K]

        # If we never recorded any pattern of this length K, return [].
        if song_id == -1 or start_idx == -1:
            return []

        song_str = self.sequences[song_id]

        # Slice out the actual notes from that song.
        # This slice is length K, so O(K).
        pattern_str = song_str[start_idx : start_idx + K]

        # Convert to list of characters, still O(K).
        return [ch for ch in pattern_str]


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