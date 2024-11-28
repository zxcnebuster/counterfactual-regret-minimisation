import operator
import os
import re
import pickle
from collections import Counter, defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional

import matplotlib.cm
import numpy as np
import pyspiel
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from tqdm import tqdm

from cfrainbow import rm
from cfrainbow.cfr import *
from cfrainbow.main import run
from cfrainbow.utils import load_game


def plot_cfr_convergence(
    iters_run: int,
    algorithm_to_expl_lists: Dict[str, List[List[float]]],
    game: Optional[pyspiel.Game] = None,
    save: bool = False,
    save_name: Optional[str] = None,
):
    with plt.style.context("bmh"):
        max_iters = 0

        # Define fixed color map for specific algorithms
        algo_colors = {
            "Vanilla CFR": "blue",
            "CFR (Alt.)": "orange",
            "DCFR": "purple",
            "DCFR (Alt.)": "brown",
            "CFR+": "green",
            # "CFR+ (Alt.)": "red",
            # "CFR-CS": "cyan",
            # "CFR-CS (Alt.)": "magenta",
            # "CFR-OS": "yellow",
            "CFR+ (QWA)": "pink",
            "MCCFR-ES":"black",
            "MCCFR-OS":"yellow"
        }

        # Ensure algorithms are always in the same order
        algo_order = [
            "CFR",
            "CFR (Alt.)",
            "DCFR",
            "DCFR (Alt.)",
            "CFR+",
            # "CFR+ (Alt.)",
            # "CFR-CS",
            # "CFR-CS (Alt.)",
            # "CFR-OS",
            "CFR+ (QWA)",
            "MCCFR-ES",
            "MCCFR-ES(None Alt)",
            "MCCFR-OS"
        ]

        # Sort the algorithms according to the fixed order
        algorithm_to_expl_lists = {
            k: algorithm_to_expl_lists[k]
            for k in algo_order if k in algorithm_to_expl_lists
        }

        # dash_pattern = matplotlib.rcParams["lines.dashed_pattern"]
        linewidth = 1.0
        x = np.arange(1, iters_run + 1)
        fig, ax = plt.subplots(figsize=(8, 6))
        manual_legend_handles = []

        for algo in algo_order:
            if algo in algorithm_to_expl_lists:
                expl_list = algorithm_to_expl_lists[algo]
                max_iters = max(max_iters, len(expl_list))
                linestyle = "--" if "(Alt.)" in algo else "-"
                color = algo_colors.get(algo, "black")  # Default to black if not specified

                if len(expl_list) != 1:
                    iteration_counts = np.flip(
                        np.sort(np.asarray([len(vals) for vals in expl_list])), 0
                    )
                    iter_beginnings = iters_run - iteration_counts
                    freq_arr = np.asarray(
                        sorted(
                            Counter(iter_beginnings).items(), key=operator.itemgetter(0)
                        ),
                        dtype=float,
                    )
                    absolute_freq = np.cumsum(freq_arr, axis=0)[:, 1]
                    relative_freq = absolute_freq / len(expl_list)
                    iter_buckets = freq_arr[:, 0].astype(int)

                    iter_to_bucket = (
                        np.searchsorted(
                            iter_buckets,
                            x - 1,
                            side="right",
                        )
                        - 1
                    )
                    no_value_mask = iter_to_bucket == -1
                    no_value_iters = x[no_value_mask]

                    absolute_freq_per_iter = np.concatenate(
                        [
                            np.zeros(no_value_iters.size),
                            absolute_freq[iter_to_bucket[~no_value_mask]],
                        ]
                    )
                    relative_freq_per_iter = np.concatenate(
                        [
                            np.zeros(no_value_iters.size),
                            relative_freq[iter_to_bucket[~no_value_mask]],
                        ]
                    )

                    padded_iter_values_matrix = (
                        torch.nn.utils.rnn.pad_sequence(
                            [torch.zeros(iters_run)] + [torch.Tensor(list(reversed(a))) for a in expl_list],
                            batch_first=True,
                        )
                        .flip(dims=(1,))
                        .numpy()
                    )
                    nanpadded_iter_values_matrix = np.where(
                        padded_iter_values_matrix == 0, np.nan, padded_iter_values_matrix
                    )
                    sum_band = padded_iter_values_matrix.sum(axis=0)
                    mean_band = sum_band / absolute_freq_per_iter
                    stderr = np.nanstd(
                        nanpadded_iter_values_matrix, ddof=1, axis=0
                    ) / np.sqrt(np.isnan(nanpadded_iter_values_matrix).sum(axis=0))

                    nan_mask = np.isnan(mean_band)
                    x_to_plot = (x - 1)[~nan_mask] + 1
                    mean_to_plot = mean_band[~nan_mask]
                    stderr_to_plot = stderr[~nan_mask]
                    points = np.array([x_to_plot, mean_to_plot]).T.reshape(-1, 1, 2)
                    segments = np.concatenate([points[:-1], points[1:]], axis=1)

                    if linestyle == "--":
                        lc = LineCollection(
                            [s for i, s in enumerate(segments) if i % 108 > 40],
                            linewidths=relative_freq_per_iter * linewidth,
                            color=color,
                            alpha=relative_freq_per_iter,
                        )
                        ax.add_collection(lc)
                    else:
                        lc = LineCollection(
                            segments,
                            linestyles=linestyle,
                            linewidths=relative_freq_per_iter * linewidth,
                            color=color,
                            alpha=relative_freq_per_iter**2,
                        )
                        ax.add_collection(lc)

                    line = Line2D(
                        [0],
                        [0],
                        label=algo,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        color=color,
                    )
                    manual_legend_handles.append(line)
                    ax.fill_between(
                        x_to_plot,
                        mean_to_plot - stderr_to_plot,
                        mean_to_plot + stderr_to_plot,
                        color=color,
                        alpha=0.1,
                    )

        for algo in algo_order:
            if algo in algorithm_to_expl_lists:
                expl_list = algorithm_to_expl_lists[algo]
                color = algo_colors.get(algo, "black")
                linestyle = "--" if "(Alt.)" in algo else "-"
                if len(expl_list) == 1:
                    ax.plot(
                        x[iters_run - len(expl_list[0]):],
                        expl_list[0],
                        label=algo,
                        linewidth=linewidth,
                        linestyle=linestyle,
                        color=color,
                    )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Exploitability")
        ax.set_yscale("log", base=10)
        ax.set_title(
            f"Convergence to Nash Equilibrium in Kuhn Poker"
        )
        manual_legend_handles.extend(ax.get_legend_handles_labels()[0])
        ax.legend(
            handles=manual_legend_handles,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fancybox=True,
            shadow=True,
            ncol=1,
        )

        if not save:
            plt.show(bbox_inches="tight")
        else:
            plt.savefig(f"./plots/plot_iters_{iters_run}.pdf", bbox_inches="tight")



def running_mean(values, window_size: int = 10):
    # return np.convolve(values, np.ones(window_size) / window_size, mode="valid")
    return [
        np.mean(values[max(0, -window_size + i) : i + 1]) for i in range(len(values))
    ]


def run_wrapper(args):
    name, pos_args, kwargs = args
    solver, exploitability = run(*pos_args, **kwargs)
    return name, exploitability, solver.average_policy()


def augment_stochastic_runs(job_list):
    augmented_jobs = []
    for alg, args in job_list:
        kwargs = args[-1]
        if kwargs.pop("stochastic_solver", False):
            augmented_jobs.extend(
                [
                    (alg, args[:-1] + (kwargs | {"seed": seed},))
                    for seed in rng.integers(0, int(1e6), size=stochastic_seeds)
                ]
            )
        else:
            augmented_jobs.append((alg, args))
    return augmented_jobs


if __name__ == "__main__":
    n_iters = 100000
    verbose = True
    game = load_game("kuhn_poker")
#     CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
# GAMEDEF
# limit
# numPlayers = 2
# numRounds = 2
# blind = 5 0
# raiseSize = 10 20 20
# maxRaises = 6 6 
# numSuits = 2
# numRanks = 6
# numHoleCards = 1
# numBoardCards = 0 1
# stack = 100
# END GAMEDEF
# """
#     game = pyspiel.universal_poker.load_universal_poker_from_acpc_gamedef(
#         CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF
#     )
    rng = np.random.default_rng(0)
    stochastic_seeds = 10

    jobs = augment_stochastic_runs(
        list(
            {
                "Vanilla CFR": (
                    VanillaCFR,
                    n_iters,
                    {
                        "game": game,
                        "regret_minimizer": rm.RegretMatcher,
                        "alternating": False,
                        "do_print": verbose,
                    },
                ),
                # "CFR (Alt.)": (
                #     VanillaCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcher,
                #         "alternating": True,
                #         "do_print": verbose,
                #     },
                # ),
                # "DCFR": (
                #     DiscountedCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcherDiscounted,
                #         "alternating": False,
                #         "do_print": verbose,
                #     },
                # ),
                "DCFR": (
                    DiscountedCFR,
                    n_iters,
                    {
                        "game": game,
                        "regret_minimizer": rm.RegretMatcherDiscounted,
                        "alternating": True,
                        "do_print": verbose,
                    },
                ),
                "CFR+": (
                    PlusCFR,
                    n_iters,
                    {
                        "game": game,
                        "regret_minimizer": rm.RegretMatcherPlus,
                        "alternating": False,
                        "do_print": verbose,
                    },
                ),
                # "CFR+ (Alt.)": (
                #     PlusCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcherPlus,
                #         "alternating": True,
                #         "do_print": verbose,
                #     },
                # ),
                # "CFR-CS": (
                #     ChanceSamplingCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcher,
                #         "alternating": False,
                #         "do_print": verbose,
                #     },
                # ),
                # "CFR-CS (Alt.)": (
                #     ChanceSamplingCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcher,
                #         "alternating": True,
                #         "do_print": verbose,
                #     },
                # ),
                # "CFR-OS (Alt.)": (
                #     OutcomeSamplingMCCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcher,
                #         "alternating": True,
                #         "do_print": verbose,
                #     },
                # ),
                # "CFR-OS": (
                #     OutcomeSamplingMCCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcher,
                #         "alternating": False,
                #         "do_print": verbose,
                #     },
                # ),
                "MCCFR-OS": (
                    OutcomeSamplingMCCFR,
                    n_iters,
                    {
                        "game": game,
                        "regret_minimizer": rm.RegretMatcher,
                        "alternating": True,
                        "do_print": verbose,
                    },
                ),
                # "MCCFR-ES(None Alt)": (
                #     ExternalSamplingMCCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.RegretMatcher,
                #         "alternating": False,
                #         "do_print": verbose,
                #     },
                # ),
                #  "PCFR+": (
                #     PredictivePlusCFR,
                #     n_iters,
                #     {
                #         "game": game,
                #         "regret_minimizer": rm.AutoPredictiveRegretMatcherPlus,
                #         "alternating": True,
                #         "do_print": verbose,
                #     },
                # ),

            }.items()
        )
    )

    n_cpu = 7
    filename = "plots/kuhn_plot_data.pkl"
    filename2 = "plots/separate_big_leduc.pkl"
    expl_dict = defaultdict(list)
    expl2_dict = defaultdict(list)
    existing_pickles = set(os.listdir("./pickles"))

    jobs_to_run = []
    run_jobs = True
    for name, values in expl_dict.items():
        print(f"{name}: {type(values)}")
        if isinstance(values, list) and values:
            if isinstance(values[0], list) and all(isinstance(x, (int, float)) for x in values[0]):
                print(f"{name} data is correct")
            else:
                print(f"Error with {name} data: Expected a list of numeric lists.")

    # for name, args in jobs:
    #     dirname = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
    #     pickle_path = f"./pickles/{dirname}.pkl"
    #     # if f"{dirname}.pkl" not in existing_pickles:
    #     jobs_to_run.append((name, args))
            
        # else:
        #     with open(pickle_path, "rb") as file:
        #         run_jobs = False
        #         policy = pickle.load(file)
        #         # Assuming the exploitability data is available in the pickle
        #         # Adjust accordingly if the data format is different
        #         expl_dict[name].append(policy)  # Or any other data stored

    if len(jobs_to_run) > 0:
        with Pool(processes=n_cpu) as pool:
            results = pool.imap_unordered(
                run_wrapper,
                (
                    (name, args_and_kwargs[:-1], args_and_kwargs[-1])
                    for (name, args_and_kwargs) in jobs_to_run
                ),
            )
            with tqdm(
                total=len(jobs_to_run),
                desc=f"Running CFR variants in multiprocess on {n_cpu} cpus",
            ) as pbar:
                for result in results:
                    pbar.update()
                    name, values, policy = result
                    expl_dict[name].append(values)
                    dirname = re.sub(r'[^a-zA-Z0-9]', '', name.lower())
                    with open(os.path.join("./pickles/", dirname+'.pkl'), "wb") as file:
                        pickle.dump(policy, file)

        with open(os.path.join(".", filename), "wb") as file:
            pickle.dump(expl_dict, file)
    else:
        if run_jobs:
            pass
            with open(os.path.join(".", filename), "rb") as file:
                expl_dict = pickle.load(file)
            #     with open(os.path.join(".", filename2), "rb") as file:
            #         expl2_dict = pickle.load(file)
            #         if 'CFR+ (QWA)'in expl2_dict:
            #             print('operation success')
            #             expl_dict['CFR+ (QWA)'] = expl2_dict.pop('CFR+ (QWA)')
            #             expl_dict['MCCFR-OS'] = expl2_dict.pop('MCCFR-OS')
                        # with open(os.path.join(".", filename), "wb") as file:
                        #     pickle.dump(expl_dict, file)

    # Debugging: Inspect the structure of expl_dict
    # for name, values in expl_dict.items():
    #     print(f"{name}: {type(values)}")
    #     if isinstance(values, list) and values:
    #         print(f"{name} first element: {type(values[0])}")

    averaged_values = {
        name: [running_mean(values, window_size=100) for values in expl_dict[name]]
        for name in expl_dict.keys()
    }
    plot_cfr_convergence(
        n_iters,
        averaged_values,
        game=game,
        save=True,
        save_name=f"{str(game)}",
    )
