import regex as re
import pickle

from cs336_basics.bpe_impl import string_to_tuple
from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.reverse_lookup = {}
        for (key, value) in self.vocab.items():
            self.reverse_lookup[value] = key
        if special_tokens:
            self.special_tokens = special_tokens
            for special in self.special_tokens:
                if special.encode("utf-8") not in self.reverse_lookup:
                    self.vocab[len(self.vocab)] = special.encode("utf-8")
                    self.reverse_lookup[special.encode("utf-8")] = len(self.reverse_lookup)
        else:
            self.special_tokens = []

    @classmethod
    def from_files(cls, vocab_file, merges_file, special_tokens=None):
        with open(vocab_file, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_file, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    
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
                    word =  string_to_tuple(pre)
                    for pair in self.merges:
                        word = self.merge_one(word, pair)
                    for part in word:
                        ending_list.append(self.reverse_lookup[part])
        return ending_list

    def decode(self, ids:list[int]) -> str:
        byte_list = []
        for id in ids:
            byte_list.append(self.vocab[id])
        return b"".join(byte_list).decode("utf-8", errors="ignore")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            for id in self.encode(string):
                yield id


tok = Tokenizer.from_files("data/tinystories_vocab.pkl", "data/tinystories_merges.pkl", ["<|endoftext|>"])
print(len(tok.vocab))
print(tok.encode("Once upon a time <|endoftext|> hi there <|endoftext|>"))
