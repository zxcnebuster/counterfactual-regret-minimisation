import operator
import pickle
import re
import os
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count

import matplotlib.cm
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tqdm import tqdm

from regretMatching import VanillaCFRRegretMatcher, DiscountedCFRRegretMatcher
from vanillaCFR import VanillaCFR
from discountedCFR import DiscountedCFR
from plusCFR import PlusCFR
from main import run
from utils import load_game

def plot_cfr_convergence(iters_run, algorithm_to_expl_lists, game=None, save=False, save_name=None):
    with plt.style.context("bmh"):
        cmap = matplotlib.cm.get_cmap("tab20")
        dash_pattern = matplotlib.rcParams["lines.dashed_pattern"]
        linewidth = 1.0
        x = np.arange(1, iters_run + 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        manual_legend_handles = []

        algorithm_to_expl_lists = sorted(algorithm_to_expl_lists.items(), key=operator.itemgetter(0))
        for i, (algo, expl_list) in enumerate(algorithm_to_expl_lists):
            color = cmap(i)
            linestyle = "--" if "(S)" in algo else "-"
            if len(expl_list) != 1:
                padded_iter_values_matrix = torch.nn.utils.rnn.pad_sequence(
                    [torch.zeros(iters_run)] + [torch.Tensor(list(reversed(a))) for a in expl_list],
                    batch_first=True,
                ).flip(dims=(1,)).numpy()
                nanpadded_iter_values_matrix = np.where(padded_iter_values_matrix == 0, np.nan, padded_iter_values_matrix)
                sum_band = padded_iter_values_matrix.sum(axis=0)
                mean_band = sum_band / np.count_nonzero(~np.isnan(nanpadded_iter_values_matrix), axis=0)
                stderr = np.nanstd(nanpadded_iter_values_matrix, ddof=1, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(nanpadded_iter_values_matrix), axis=0))
                nan_mask = np.isnan(mean_band)
                x_to_plot = x[~nan_mask]
                mean_to_plot = mean_band[~nan_mask]
                stderr_to_plot = stderr[~nan_mask]
                points = np.array([x_to_plot, mean_to_plot]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, linestyles=linestyle, linewidths=linewidth, color=color, alpha=0.7)
                ax.add_collection(lc)
                line = Line2D([0], [0], label=algo, linewidth=linewidth, linestyle=linestyle, color=color)
                manual_legend_handles.append(line)
                ax.fill_between(x_to_plot, mean_to_plot - stderr_to_plot, mean_to_plot + stderr_to_plot, color=color, alpha=0.1)
            else:
                ax.plot(x[iters_run - len(expl_list[0]):], expl_list[0], label=algo, linewidth=linewidth, linestyle=linestyle, color=color)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Exploitability")
        ax.set_yscale("log", base=10)
        ax.set_title(f"Convergence to Nash Equilibrium {'in ' + str(game) if game is not None else ''}")
        ax.legend(handles=manual_legend_handles, loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)

        if save:
            plt.savefig(f"./plots/plot_iters_{iters_run}.pdf", bbox_inches="tight")
        else:
            plt.show(bbox_inches="tight")

        # Plotting the last 10% of the iteration data with higher precision
        fig, ax = plt.subplots(figsize=(8, 6))
        last_10_percent = int(iters_run * 0.1)
        x_last_10 = np.arange(iters_run - last_10_percent + 1, iters_run + 1)
        
        for i, (algo, expl_list) in enumerate(algorithm_to_expl_lists):
            color = cmap(i)
            linestyle = "--" if "(S)" in algo else "-"
            if len(expl_list) != 1:
                padded_iter_values_matrix = torch.nn.utils.rnn.pad_sequence(
                    [torch.zeros(iters_run)] + [torch.Tensor(list(reversed(a))) for a in expl_list],
                    batch_first=True,
                ).flip(dims=(1,)).numpy()
                nanpadded_iter_values_matrix = np.where(padded_iter_values_matrix == 0, np.nan, padded_iter_values_matrix)
                sum_band = padded_iter_values_matrix[:, -last_10_percent:].sum(axis=0)
                mean_band = sum_band / np.count_nonzero(~np.isnan(nanpadded_iter_values_matrix[:, -last_10_percent:]), axis=0)
                stderr = np.nanstd(nanpadded_iter_values_matrix[:, -last_10_percent:], ddof=1, axis=0) / np.sqrt(np.count_nonzero(~np.isnan(nanpadded_iter_values_matrix[:, -last_10_percent:]), axis=0))
                nan_mask = np.isnan(mean_band)
                x_to_plot = x_last_10[~nan_mask]
                mean_to_plot = mean_band[~nan_mask]
                stderr_to_plot = stderr[~nan_mask]
                points = np.array([x_to_plot, mean_to_plot]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                lc = LineCollection(segments, linestyles=linestyle, linewidths=linewidth, color=color, alpha=0.7)
                ax.add_collection(lc)
                line = Line2D([0], [0], label=algo, linewidth=linewidth, linestyle=linestyle, color=color)
                manual_legend_handles.append(line)
                ax.fill_between(x_to_plot, mean_to_plot - stderr_to_plot, mean_to_plot + stderr_to_plot, color=color, alpha=0.1)
            else:
                ax.plot(x_last_10, expl_list[0][-last_10_percent:], label=algo, linewidth=linewidth, linestyle=linestyle, color=color)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Exploitability")
        ax.set_title(f"Last 10% Convergence to Nash Equilibrium {'in ' + str(game) if game is not None else ''}")
        ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
        ax.legend(handles=manual_legend_handles, loc="center left", bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True, ncol=1)

        if save:
            plt.savefig(f"./plots/plot_iters_{iters_run}_last_10_percent.pdf", bbox_inches="tight")
        else:
            plt.show(bbox_inches="tight")

def running_mean(values, window_size=10):
    return [np.mean(values[max(0, -window_size + i): i + 1]) for i in range(len(values))]

def run_wrapper(args):
    name, pos_args, kwargs = args
    solver, exploitability = run(*pos_args, **kwargs)
    return name, exploitability, solver.get_average_strategy()

def augment_stochastic_runs(job_list):
    augmented_jobs = []
    for alg, args in job_list:
        kwargs = args[-1]
        if kwargs.pop("stochastic_solver", False):
            augmented_jobs.extend([(alg, args[:-1] + (kwargs | {"seed": seed},)) for seed in rng.integers(0, int(1e6), size=stochastic_seeds)])
        else:
            augmented_jobs.append((alg, args))
    return augmented_jobs

if __name__ == "__main__":
    n_iters = 10000
    game = load_game("leduc_poker")
    rng = np.random.default_rng(0)
    stochastic_seeds = 10

    jobs = augment_stochastic_runs(
        list(
            {
                "CFR": (VanillaCFR, n_iters, {"game": game, "regret_minimizer": VanillaCFRRegretMatcher, "alternating": True}),
                "DCFR": (DiscountedCFR, n_iters, {"game": game, "regret_minimizer": DiscountedCFRRegretMatcher, "alternating": True}),
                "CFR+": (PlusCFR, n_iters, {"game": game, "regret_minimizer": DiscountedCFRRegretMatcher, "alternating": True}),
            }.items()
        )
    )

    n_cpu = cpu_count()
    with Pool(processes=n_cpu) as pool:
        results = pool.imap_unordered(run_wrapper, ((name, args_and_kwargs[:-1], args_and_kwargs[-1]) for (name, args_and_kwargs) in jobs))
        expl_dict = defaultdict(list)
        with tqdm(total=len(jobs), desc=f"Running CFR variants in multiprocess on {n_cpu} cpus") as pbar:
            for result in results:
                pbar.update()
                name, values, policy = result
                expl_dict[name].append(values)
                dirname = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
                with open(os.path.join("./pickles/", dirname+'.pkl'), "wb") as file:
                    pickle.dump(policy, file)

    averaged_values = {name: [running_mean(values, window_size=100) for values in expl_values] for name, expl_values in expl_dict.items()}
    plot_cfr_convergence(n_iters, averaged_values, game=game, save=True, save_name=f"{str(game)}")