from bpe_impl import BPE
import resource
import time
import pickle

instance = BPE("/Users/jerryqi/Documents/Github/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])
start = time.perf_counter()
instance.train()
end = time.perf_counter()

# Save the vocab and merges so that we can reuse these for other stuff later
with open("data/tinystories_vocab.pkl", "wb") as f:
    pickle.dump(instance.vocab, f)

with open("data/tinystories_merges.pkl", "wb") as f:
    pickle.dump(instance.merges, f)



print("Time used (seconds): ", end - start)

print("Memory used (GB): ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)

