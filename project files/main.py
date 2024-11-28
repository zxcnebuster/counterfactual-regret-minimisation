import inspect

import pyspiel
from open_spiel.python.algorithms import exploitability
from tqdm import tqdm

from regretMatching import VanillaCFRRegretMatcher
from abstractCFR import AbstractCFR
from utils import (
    KuhnPolicyPrinter,
    PolicyPrinter,
    load_game,
    make_uniform_policy,
    normalize_policy_profile,
    slice_kwargs,
    to_pyspiel_policy,
)

def run(
    solver,
    n_iter,
    regret_minimizer = VanillaCFRRegretMatcher,
    game = "kuhn_poker",
    policy_printer = None,
    progressbar = False,
    final_expl_print = False,
    expl_threshold = None,
    expl_check_freq = 1,
    **kwargs,
):
    # get all kwargs that can be found in the parent classes' and the given class's __init__ func
    solver_kwargs = slice_kwargs(
        kwargs, *[cls.__init__ for cls in inspect.getmro(solver)]
    )
    do_print = policy_printer is not None
    if do_print:
        print(
            f"Running {solver.__name__} "
            f"with regret minimizer {regret_minimizer.__name__} "
            f"and kwargs {solver_kwargs} "
            f"for {n_iter} iterations."
        )

    expl_values = []
    game = load_game(game)
    root_states = game.new_initial_states()
    uniform_joint_policy = dict()
    for root_state in root_states:
        uniform_joint_policy.update(make_uniform_policy(root_state))
    n_infostates = len(uniform_joint_policy)

    solver_obj = solver(
        root_states,
        regret_minimizer,
        
        **solver_kwargs,
    )

    gen = range(n_iter)
    for iteration in gen if not progressbar else (pbar := tqdm(gen)):
        if progressbar:
            if expl_values:
                expl_print = f"{f'{expl_values[-1]: .5f}' if expl_values and expl_values[-1] > 1e-5 else f'{expl_values[-1]: .3E}'}"
            else:
                expl_print = " - "
            pbar.set_description(
                f"Method:{solver.__name__} | "
                f"RM:{regret_minimizer.__name__} | "
                f"kwargs:{solver_kwargs} | "
                f"{expl_print}"
            )

        solver_obj.iterate()

        avg_policy = solver_obj.get_average_strategy()
        if iteration % expl_check_freq == 0:
            expl_values.append(
                exploitability.exploitability(
                    game,
                    to_pyspiel_policy(avg_policy, uniform_joint_policy),
                )
            )

            if do_print or (iteration == n_iter - 1 and final_expl_print):
                printed_profile = policy_printer.print_profile(
                    normalize_policy_profile(avg_policy)
                )
                if printed_profile:
                    print(
                        f"-------------------------------------------------------------"
                        f"--> Exploitability "
                        f"{f'{expl_values[-1]: .5f}' if 1e-5 < expl_values[-1] < 1e7 else f'{expl_values[-1]: .3E}'}"
                    )
                    print(printed_profile)
                    print(
                        f"---------------------------------------------------------------"
                    )

            if expl_threshold is not None and expl_values[-1] < expl_threshold:
                print(f"Exploitability threshold of {expl_threshold} reached.")
                break
    if do_print:
        print(
            "\n---------------------------------------------------------------> Final Exploitability:",
            expl_values[-1],
        )
    avg_policy = solver_obj.get_average_strategy()
    if (do_print or final_expl_print) and sum(
        map(lambda p: len(p), avg_policy)
    ) == n_infostates:
        print(policy_printer.print_profile(solver_obj.get_average_strategy()))

    return solver_obj, expl_values