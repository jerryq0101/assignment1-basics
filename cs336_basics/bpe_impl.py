# Stupid Implementation first
# then do the smart implementation

import collections
import regex as re
import time
import pickle

class BPE:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.desired_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.merges = []
        self.vocab = {}
        self.reverse_lookup = {}
        # Initialize the vocabulary with the 0-255 elements
        for i in range(256):
            self.vocab[i] = bytes([i])
            self.reverse_lookup[bytes([i])] = i
        # Initialize special tokens
        for tok in special_tokens:
            self.vocab[len(self.vocab)] = bytes(tok.encode("utf-8"))
            self.reverse_lookup[bytes(tok.encode("utf-8"))] = len(self.reverse_lookup)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens):
        self = cls.__new__(cls)              # allocate without running __init__
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
        with open(merges_path, "rb") as f:
            self.merges = pickle.load(f)
        self.special_tokens = special_tokens
        self.input_path = None
        self.reverse_lookup = {b: i for i, b in self.vocab.items()}
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        return self
        
    def build_special_pattern(self):
        # Process special tokens
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        processed_special_tokens = []
        for element in self.special_tokens:
            # go through all the special tokens individually to see if there are any special tokens
            processed_special_tokens.append(re.escape(element))
        str_expr = "|".join(processed_special_tokens)
        # And also isolate special_tokens by themselves.
        return str_expr

    def regex_pretokenize(self, segment):
        # Returns an iterator of pre-token strings
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        return (m.group() for m in re.finditer(PAT, segment))

    def pre_tokenization(self, input: str) -> list[str]:
        # Split using some sort of regex into a array of strings
        # Deal with special tokens somehow 

        str_arr = [input]

        # Dumb implementation of the actual thing
        if len(self.special_tokens) != 0:
            special_pattern = self.build_special_pattern()
            str_arr = re.split(special_pattern, input)

        # Split using the regex on each smaller section
        # Optimization for RAM
        counts = collections.Counter()
        for seg in str_arr:
            counts.update(self.regex_pretokenize(seg))
        
        byte_based_counts = {}
        for (word, count) in counts.items():
            word_in_utf8 = word.encode("utf-8")
            tuple_with_bytes = tuple(bytes([c]) for c in word_in_utf8)
            byte_based_counts[tuple_with_bytes] = count
        return byte_based_counts


    def merge_one_step(self, byte_based_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
        # Get the counts of the most common occurances in bytes
        # Then merge the most common one
        # Add that to vocabulary, and merges

        # Implementation
        # Imagine get_counts returned something like (e,s): 3, ...
        counts = self.get_counts(byte_based_counts)
        pair_to_merge = self.select_best(counts)
        if len(counts) == 0:
            return None
        self.merges.append(pair_to_merge)
        self.vocab[len(self.vocab)] = pair_to_merge[0] + pair_to_merge[1]
        self.reverse_lookup[pair_to_merge[0] + pair_to_merge[1]] = len(self.reverse_lookup)
        post_merge_byte_based_counts = {}
        for (word, count) in byte_based_counts.items():
            rewritten_word = self.merge_one(word, pair_to_merge)
            post_merge_byte_based_counts[rewritten_word] = count
        
        return post_merge_byte_based_counts

    # Merge particular pair inside of the word
    def merge_one(self, word, pair):
        i = 0
        out = []
        while i < len(word):
            if i+1 < len(word) and (word[i] == pair[0] and word[i+1] == pair[1]):
                combined = word[i] + word[i+1]
                out.append(combined)
                i+=2
            else:
                out.append(word[i])
                i+=1
        rewritten_word = tuple(out)
        return rewritten_word

    def select_best(self, counts: dict[tuple[bytes], int]):
        best = ()
        best_count = 0
        if len(counts) == 0:
            return None
        for (pair, count) in counts.items():
            if count > best_count or (count == best_count and pair > best):
                best = pair
                best_count = count
        return best

        
    def get_counts(self, byte_based_counts: dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
        # Get counts
        global_dict = {}
        for (tup, count) in byte_based_counts.items():
            i = 0
            while i+1 < len(tup):
                pair = (tup[i], tup[i+1])
                global_dict[pair] = global_dict.get(pair, 0) + count
                i+=1
        return global_dict
                

    def train(self):
        # get the actual string
        with open(self.input_path, encoding="utf-8") as f:
            text = f.read()
            # Pretokenize
            t0 = time.perf_counter()
            preprocess = self.pre_tokenization(text)
            t1 = time.perf_counter()
            # Train the particular tokenizer
            while len(self.vocab) < self.desired_vocab_size:
                temp = self.merge_one_step(preprocess)
                if temp is None:
                    break
                preprocess = temp
            t2 = time.perf_counter()
            print("T0: ", t0)
            print("T1: ", t1)
            print("T2: ", t2)
            return preprocess

    def encode_iterable(self, iterable):
        for chunk in iterable:              # e.g. a file handle → one line at a time
            yield from self.encode(chunk)

    def encode(self, input: str) -> list[bytes]:
        # Pretokenize into a list
        # Convert it to UTF8
        # Iterate through merges
            # For all the input, we would get all the pairs that match the merges element, and then merge it according to it
        # Return the resulting
        pattern_to_split_on = "(" + self.build_special_pattern() + ")"
        input_split = re.split(pattern_to_split_on, input)
        if not self.special_tokens:
            input_split = [input]

        ending_list = []
        for some_text in input_split:
            if some_text in self.special_tokens:
                ending_list.append(self.reverse_lookup[some_text.encode("utf-8")])
            else:
                for pre in self.regex_pretokenize(some_text):
                    word = string_to_tuple(pre)
                    while len(word) > 1:
                        pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
                        best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
                        if best not in self.ranks:
                            break
                        word = self.merge_one(word, best)

                    # Convert into list of encoded tokens
                    for part in word:
                        ending_list.append(self.reverse_lookup[part])
        return ending_list
        


    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")


def string_to_tuple(input: str):
    byte_form = input.encode("utf-8")
    return tuple(bytes([c]) for c in byte_form)

