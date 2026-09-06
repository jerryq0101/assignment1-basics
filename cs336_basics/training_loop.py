import torch.nn
import math
import os
import numpy as np
import pickle
import time
from cs336_basics.transformer_lm import TransformerLMModule
from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import CheckpointingModule
from cs336_basics.cosine_lr import CosineLearningRateSchedule
from cs336_basics.data_loading import DataLoadingModule
from cs336_basics.cross_entropy_loss import CrossEntropyModule
from cs336_basics.gradient_clipping import GradientClipping
from cs336_basics.bpe_impl import BPE
from cs336_basics.training_loop_graphing import TrainingStatsUtil



class TrainingModule():
    def __init__(self):
        pass

    def train_something(self, experiment_num: int, desired_device: str, hyperparam_updates: dict):
        # --- data ---
        print(torch.__version__)
        train_path     = "data/tinystories_train.npy"
        val_path       = "data/tinystories_val.npy"
        device         = "mps" if torch.backends.mps.is_available() else "cpu"
        if desired_device  == "cuda" and torch.cuda.is_available():
            device = "cuda"
        print("Chosen device: ", device)

        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs("exp_logs", exist_ok=True)

        # Make experiment num file if it doesn't exist yet
        log_fd = open(f'exp_logs/log_{experiment_num}', "a")
        time_start = time.time()
        existing_time = 0

        hyperparams = {
            # --- model ---
            "vocab_size"     : 10000,
            "context_length" : 256,
            "d_model"        : 512,
            "d_ff"           : 1344,          # ≈ (8/3)·d_model rounded to a multiple of 64,
            "num_layers"     : 4,
            "num_heads"      : 16,
            "rope_theta"     : 10000.0,
            "rmsnorm_eps"    : 1e-5,

            # --- optimizer (AdamW) ---
            "lr_max"         : 1e-3,
            "lr_min"         : 1e-4,
            "weight_decay"   : 0.01,
            "betas"          : (0.9, 0.95),
            "eps"            : 1e-8,
            "max_grad_norm"  : 1.0,

            # --- schedule / run length ---
            "max_iters"      : 5000,
            "warmup_iters"   : 1000,
            "cosine_iters"   : 5000,     # T_c; usually the full run

            # --- batching ---
            "batch_size"     : 32,

            # --- logging / eval ---
            "log_every"      : 50,
            "eval_every"     : 50,
            "eval_batches"   : 20            # batches averaged per val-loss estimate
        }
        self.context_length = hyperparams["context_length"]

        train_data = np.load(train_path, mmap_mode="r")
        val_data = np.load(val_path, mmap_mode="r")
        
        checkpoint_src = None
        if os.path.exists(f"checkpoints/ckpt_exp{experiment_num}_latest.pt"):
            checkpoint_src = f"checkpoints/ckpt_exp{experiment_num}_latest.pt"

        # Can construct first on CPU
        #  and then let .to(device) sweep the whole registered tree across
        # This works for MPS
        # For a discrete GPU, doing initialization first in CPU will take up DRAM
        # Then it will transfer to VRAM. So it might OOO because CPU DRARM might be too small
        model = TransformerLMModule(vocab_size=hyperparams["vocab_size"], 
                                    context_length=hyperparams["context_length"],
                                    d_model=hyperparams["d_model"],
                                    num_layers=hyperparams["num_layers"],
                                    num_heads=hyperparams["num_heads"],
                                    d_ff=hyperparams["d_ff"],
                                    rope_theta=hyperparams["rope_theta"],
                                    eps=hyperparams["rmsnorm_eps"]).to(device)
        self.model = model
        print("Get shape of state dict: ", self.model.state_dict().keys())
        opt = AdamW(params=model.parameters(), lr=hyperparams["lr_max"], weight_decay=hyperparams["weight_decay"], betas=hyperparams["betas"], eps=hyperparams["eps"])
        
        start = 0
        checkpoint_util = CheckpointingModule()

        # Load the checkpoint
        if checkpoint_src is not None:
            (ckpt_start, ckpt_hyper, training_time_so_far) = checkpoint_util.load_checkpoint(src=checkpoint_src, model=model, optimizer=opt)
            hyperparams = ckpt_hyper
            start = ckpt_start
            existing_time = training_time_so_far
            print("Loaded from checkpoint! Starting iteration at: ", start)
            assert(start is not None)
        # Update hyperparameters if there are fields that we want to change
        if hyperparam_updates is not None:
            hyperparams.update(hyperparam_updates)


        lr_util = CosineLearningRateSchedule()
        data_loading_util = DataLoadingModule()
        cross_entropy_util = CrossEntropyModule()
        grad_clipping_util = GradientClipping()

        best_eval_loss = float("inf")

        for it in range(start, hyperparams["max_iters"]):
            lr = lr_util.get_rate(t=it, a_max=hyperparams["lr_max"], a_min=hyperparams["lr_min"], T_w=hyperparams["warmup_iters"], T_c=hyperparams["cosine_iters"])
            # Feed the optimizer on these rates for this step
            for g in opt.param_groups:
                g["lr"] = lr

            x, y = data_loading_util.get_batch(x=train_data, batch_size=hyperparams["batch_size"], context_length=hyperparams["context_length"], device=device)
            # Forward and then get the loss
            res = model(x)
            loss = cross_entropy_util(res, y)

            # Zero gradients and then backward pass
            opt.zero_grad()
            loss.backward()

            # Clip gradients to prevent craziness
            grad_clipping_util.clip(model.parameters(), max_l2_norm=hyperparams["max_grad_norm"])
            
            # Do an optimizer step
            opt.step()

            if it % hyperparams["log_every"] == 0:
                print("Train loss: ", loss.item(), " LR: ", lr)

            @torch.no_grad()
            def estimate_val_loss():
                total = 0.0
                for _ in range(hyperparams["eval_batches"]):
                    xv, yv = data_loading_util.get_batch(val_data, hyperparams["batch_size"], hyperparams["context_length"], device)
                    total += cross_entropy_util(model(xv), yv).item()
                return total / hyperparams["eval_batches"]

            if it % hyperparams["eval_every"] == 0:
                eval_loss_curr = estimate_val_loss()
                print(f"step {it}  val loss {eval_loss_curr:.4f}")
                print("\nSaving checkpoint\n")
                time_interval = time.time() - time_start 
                new_accumulated_time = existing_time + time_interval
                existing_time = new_accumulated_time
                time_start = time.time()
                if eval_loss_curr < best_eval_loss:
                    checkpoint_util.save_checkpoint(model=model, optimizer=opt, iteration=it, hyper=hyperparams,out=f"checkpoints/ckpt_exp{experiment_num}_best_eval.pt", training_time=new_accumulated_time)
                    best_eval_loss = eval_loss_curr
                checkpoint_util.save_checkpoint(model=model, optimizer=opt, iteration=it, hyper=hyperparams,out=f"checkpoints/ckpt_exp{experiment_num}_latest.pt", training_time=new_accumulated_time)
                print("\nFinished saving checkpoint\n")
                print("Saving logs: \n")
                log_fd.write(f"{it}, {eval_loss_curr}, {new_accumulated_time}\n")
                log_fd.flush()
                print("Finished saving logs")

        log_fd.close()

        # Graph this specific train as we are done
        TrainingStatsUtil().graph_loss_curve(f"exp_logs/log_{experiment_num}", experiment_num=experiment_num)


    @torch.no_grad()
    def generate(self, tokenizer: BPE, prompt: str, max_new_tokens: int, temperature: float = 1.0, top_p: float = 0.9):
        device = next(self.model.parameters()).device
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.int64, device=device)  # (1, T)
        eos_id = tokenizer.reverse_lookup["<|endoftext|>".encode("utf-8")]

        for _ in range(max_new_tokens):
            logits = self.model(ids[:, -self.context_length:])   # crop to trained context
            probs = torch.softmax(logits[0, -1] / temperature, dim=-1)   # last position only

            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=-1)
            sorted_probs[cumsum - sorted_probs >= top_p] = 0.0
            sorted_probs /= sorted_probs.sum()
            next_id = sorted_idx[torch.multinomial(sorted_probs, 1)]     # (1,)

            ids = torch.cat([ids, next_id.view(1, 1)], dim=1)
            if next_id.item() == eos_id:
                break
            print ("Current generated text: ", tokenizer.decode(ids[0].tolist()), "\n")

        return tokenizer.decode(ids[0].tolist())
        


if __name__ == "__main__":
    # Do construction of the tokenized input output before running this on it
    RAW = {"train": "data/TinyStoriesV2-GPT4-train.txt",
        "val":   "data/TinyStoriesV2-GPT4-valid.txt"}

    if not os.path.exists("data/tinystories_vocab.pkl"):
        tokenizer = BPE(RAW["train"], 10000, ["<|endoftext|>"])
        tokenizer.train()
        with open("data/tinystories_vocab.pkl", "wb") as f:  pickle.dump(tokenizer.vocab, f)
        with open("data/tinystories_merges.pkl", "wb") as f: pickle.dump(tokenizer.merges, f)


    tokenizer = BPE.from_files("data/tinystories_vocab.pkl", "data/tinystories_merges.pkl", ["<|endoftext|>"])

    # If the train and val encodings don't already exist, then convert
    if not os.path.exists("data/tinystories_val.npy"):
        # Encode the train and eval datasets
        for split, raw in RAW.items():
            out = f"data/tinystories_{split}.npy"
            if not os.path.exists(out):
                with open(raw, encoding="utf-8") as f:
                    ids = np.fromiter(tokenizer.encode_iterable(f), dtype=np.uint16)
                assert ids.max() < 10000, f"{split}: bad id {ids.max()}"
                np.save(out, ids)

    # Already have done tokenization, do the experiment
    # Weird one
    modelModule1 = TrainingModule()
    modelModule1.train_something(experiment_num=1, desired_device="cuda", hyperparam_updates={
        "lr_max":1e-2,
        "lr_min": 1e-5,
    })

    modelModule2 = TrainingModule()
    modelModule2.train_something(experiment_num=1, desired_device="cuda", hyperparam_updates={
        "lr_max":1e-3,
        "lr_min": 1e-4,
    })

    modelModule3 = TrainingModule()
    modelModule3.train_something(experiment_num=1, desired_device="cuda", hyperparam_updates={
        "lr_max":1e-2,
        "lr_min": 1e-3,
    })


    modelModule4 = TrainingModule()
    modelModule4.train_something(experiment_num=2, desired_device="cuda", hyperparam_updates={
        "lr_max":1e-1,
        "lr_min":1e-2
    })

    modelModule5 = TrainingModule()
    modelModule5.train_something(experiment_num=2, desired_device="cuda", hyperparam_updates={
        "lr_max":1e0,
        "lr_min":1e-1
    })
