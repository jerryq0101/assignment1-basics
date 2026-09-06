import numpy as np 
import matplotlib.pyplot  as plt
import os

class TrainingStatsUtil():
    def __init__(self):
        pass

    def graph_loss_curve(self, log_path: str, experiment_num: int):
        stats = np.loadtxt(log_path, delimiter=",")
        print(stats.shape)

        os.makedirs("exp_graphs", exist_ok=True)

        iterations = stats[:, 0]
        val_losses = stats[:, 1]
        elapsed_seconds = stats[:, 2]

        # Do the loss vs steps
        plt.figure()
        plt.plot(iterations, val_losses)
        plt.xlabel("Gradient Step")
        plt.ylabel("Val Loss (Eval)")
        plt.savefig(f'exp_graphs/exp_{experiment_num}_step_loss.png')
        plt.close()

        # Loss vs time
        plt.figure()
        plt.plot(elapsed_seconds, val_losses)
        plt.xlabel("Elapsed time (sec)")
        plt.ylabel("Val losses (Eval)")
        plt.savefig(f'exp_graphs/exp_{experiment_num}_time_loss.png')
        plt.close()
