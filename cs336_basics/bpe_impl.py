# Stupid Implementation first
# then do the smart implementation

import collections
import regex as re

class BPE:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.input_path = input_path
        self.desired_vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.merges = []
        self.vocab = {}
        # Initialize the vocabulary with the 0-255 elements
        for i in range(256):
            self.vocab[i] = bytes([i])
        # Initialize special tokens
        for tok in special_tokens:
            self.vocab[len(self.vocab)] = bytes(tok.encode("utf-8"))

    def pre_tokenization(self, input: str) -> list[str]:
        # Split using some sort of regex into a array of strings
        # Deal with special tokens somehow 

        str_arr = [input]

        # Dumb implementation of the actual thing
        if len(self.special_tokens) != 0:
            # Process special tokens
            processed_special_tokens = []
            for element in self.special_tokens:
                # go through all the special tokens individually to see if there are any special tokens
                processed_special_tokens.append(re.escape(element))
            str_expr = "|".join(processed_special_tokens)
            # And also isolate special_tokens by themselves.
            str_arr = re.split(str_expr, input)

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # Split using the regex on each smaller section
        # Optimization for RAM
        counts = collections.Counter()
        for element in str_arr:
            counts.update((m.group() for m in re.finditer(PAT, element)))
        
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
            preprocess = self.pre_tokenization(text)
            # Train the particular tokenizer
            while len(self.vocab) < self.desired_vocab_size:
                temp = self.merge_one_step(preprocess)
                if temp is None:
                    break
                preprocess = temp
            return preprocess

    def encode(self, input: str) -> list[bytes]:
        # Pretokenize into a list
        # Convert it to UTF8
        # Iterate through merges
            # For all the input, we would get all the pairs that match the merges element, and then merge it according to it
        # Return the resulting
        split_str = input.split()
        # Convert each thing to utf8
        str_converted = []
        for piece in split_str:
            str_converted.append(tuple(bytes([c]) for c in piece.encode("utf-8")))

        for i in range(len(self.merges)):
            # Scan if there exists such a sequence here
            for j in range(len(str_converted)):
                str_converted[j] = self.merge_one(str_converted[j], self.merges[i])

        final_list = []
        for tup in str_converted:
            for item in tup:
                final_list.append(item)
        return final_list
        

    def decode(self, bytes: list[bytes]):
        # Use our vocabulary to convert this back into the vocabulary
        # Build up vocabs using the merges dict.
        # Then just straight up convert each thing 
        pass


def string_to_tuple(input: str):
    byte_form = input.encode("utf-8")
    return tuple(bytes([c]) for c in byte_form)

