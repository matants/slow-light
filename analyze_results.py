import ast
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sld_left_right_split_no_scatter import SlowLightDeviceWithLeftRightSplitNoScatter


def compare_to_simulations(fpath, num_of_simulations: int = 100, seed: int = 73):
    data = np.load(fpath)

    Lt = data['Lt'][-1]
    dt = data['dt'][-1]
    p_finish = data['p_finish'][-1]
    reward_conditioned = data['reward_conditioned'][-1]
    loss = data['loss'][-1]

    best_sample = data['best_sample'][-1]
    if best_sample == -1:
        raise ValueError("Bad sample")

    Lt_best = Lt[best_sample]
    dt_best = dt[best_sample]
    sld = SlowLightDeviceWithLeftRightSplitNoScatter(L_init=Lt_best, d_init=dt_best, batch_size=1, N=len(Lt_best),
                                                     seed=seed, is_optimize=False)

    experimental_rew, experimental_pf = sld.score_simulations_accumulated(num_of_simulations)
    estimated_rew = reward_conditioned[best_sample]
    estimated_pf = p_finish[best_sample]

    print(f"Estimated reward =/t{estimated_rew:.2f},/testimated p_finish =/t{estimated_pf:.4f}")
    print(f"Experimental reward =/t{experimental_rew:.2f},/texperimental_pf =/t{experimental_pf:.4f}")


def plot_best_learning_curve(fpath):
    data = np.load(fpath)
    iters = data['iter']
    reward_conditioned = data['reward_conditioned']
    loss = data['loss']
    p_finish = data['p_finish']
    best_sample = loss.argmin(axis=1)
    n_batches = loss.shape[1]
    len_iters = len(iters)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.title.set_text("A. Total loss")
    ax1.plot(iters, loss[range(len_iters), best_sample], c='b')
    ax1.set_xlabel('# iterations')
    ax1.set_ylabel('loss')
    ax1.grid()

    ax2.title.set_text("B. Finish probability")
    ax2.plot(iters, p_finish[range(len_iters), best_sample], c='r')
    ax2.set_xlabel('# iterations')
    ax2.set_ylabel('finish probability')
    ax2.grid()

    ax3.title.set_text("C. Device conditional reward")
    ax3.plot(iters, reward_conditioned[range(len_iters), best_sample], c='g')
    ax3.set_xlabel('# iterations')
    ax3.set_ylabel('reward')
    ax3.grid()


def plot_only_loss(fpath):
    data = np.load(fpath)
    iters = data['iter']
    loss = data['loss']
    best_sample = loss.argmin(axis=1)
    n_batches = loss.shape[1]
    len_iters = len(iters)
    fig, ax1 = plt.subplots(1, 1)

    # ax1.title.set_text("Loss")
    ax1.plot(iters, loss[range(len_iters), best_sample], c='b')
    ax1.set_xlabel('# iterations')
    ax1.set_ylabel('loss')
    ax1.grid()


def print_sorted_permutations_csv(fpath, num_rows=10, is_best_not_worst=True):
    df = pd.read_csv(fpath, delimiter=":", header=None, names=['permutation', 'loss'])
    df["loss"] = pd.to_numeric(df["loss"])
    sorted_df = df.sort_values(by='loss')
    print_df_for_latex(sorted_df, num_rows, is_best_not_worst)
    return sorted_df


def print_sorted_permutations_according_to_lengths(fpath, order_by: str = 'Lt', num_rows=10, is_best_not_worst=True):
    init_lengths_path = os.path.dirname(fpath) + "/saved_init_lengths.npz"
    init_lengths = np.load(init_lengths_path)
    # I use here the double argsort trick, see https://www.berkayantmen.com/rank
    order = list(np.argsort(np.argsort(np.squeeze(init_lengths[order_by]))))
    print("Original device order: {}".format(order))
    df = pd.read_csv(fpath, delimiter=":", header=None, names=['permutation', 'loss'])
    df["loss"] = pd.to_numeric(df["loss"])
    sorted_df = df.sort_values(by='loss')
    for i in range(len(sorted_df)):
        sorted_df.loc[i, 'permutation'] = convert_permutation_to_new_order(sorted_df.loc[i, 'permutation'], order)
    print_df_for_latex(sorted_df, num_rows, is_best_not_worst)
    return sorted_df


def convert_permutation_to_new_order(initial_permutation: str, new_order: list):
    permutation_tuple = ast.literal_eval(initial_permutation)
    assert len(permutation_tuple) == len(new_order), "Lengths don't match."
    return str(tuple([new_order[i] for i in permutation_tuple]))


def print_df_for_latex(df, n=10, is_best_not_worst=True):
    print_this = ""
    for i in range(n):
        if is_best_not_worst:
            row = df.iloc[i]
        else:
            row = df.iloc[len(df) - n + i]
        row_str = ""
        n_cols = len(row)
        for col, thing in enumerate(row):
            if type(thing) is np.float64 or type(thing) is float:
                thing_str = f"{thing:.4f}"
            else:
                thing_str = str(thing)
            row_str += "$" + thing_str + "$"
            if col < n_cols - 1:
                row_str += "&"
            else:
                row_str += "\\\\\n"
        print_this += row_str
    print(print_this)


def exponential_smooth(scalars: list, weight: float) -> list:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed


def extract_best_lengths(fpath):
    results = np.load(fpath)
    loss = results['loss']
    ind = np.unravel_index(np.argmin(loss, axis=None), loss.shape)
    return results['Lt'][ind], results['dt'][ind]


def extract_final_lengths(fpath):
    results = np.load(fpath)
    ind = results['loss'][-1].argmin()
    return results['Lt'][-1, ind], results['dt'][-1, ind]


def extract_best_ind(fpath):
    results = np.load(fpath)
    loss = results['loss']
    ind = np.unravel_index(np.argmin(loss, axis=None), loss.shape)
    return results, ind


def find_template_if_last(str_input, template):
    ind_of_template = str_input.find(template)
    if ind_of_template == -1:
        return False
    # Check if template is the end of the string
    if len(str_input) != ind_of_template + len(template):
        return False
    return True


def get_all_results_of_path_template(value_name, path_template, seed_range, value_range):
    df = pd.DataFrame(columns=[value_name, "seed", "loss", "p_finish", "reward_conditioned"])
    for seed in seed_range:
        for value in value_range:
            # find dir
            dir_found = None
            for dir_name in os.listdir("./runs"):
                if find_template_if_last(dir_name, path_template.format(seed, value)):
                    if dir_found is None:
                        dir_found = dir_name
                    else:
                        raise RuntimeError("Found 2 different matching dirs: {}, {}".format(dir_found, dir_name))
            if dir_found is None:
                raise RuntimeError(
                    "Found no matching dirs to the template {}".format(path_template.format(seed, value)))
            # extract data
            data_path = "./runs/" + dir_found + "/results.npz"
            results, ind = extract_best_ind(data_path)
            df = df.append({
                value_name: value,
                "seed": seed,
                "loss": results["loss"][ind],
                "p_finish": results["p_finish"][ind],
                "reward_conditioned": results["reward_conditioned"][ind],
                # "p_scatter": results["p_scatter"][ind],
                "p_reflect": results["p_reflect"][ind],
            }, ignore_index=True)
    df = df.astype({"seed": "int32"})
    return df


if __name__ == '__main__':
    fpath = './runs/run_2021_06_19__17_44__initial_results_after_fix/results.npz'
    ##############################################################
    # Plot loss, pf, reward
    ##############################################################
    # plot_best_learning_curve(fpath)

    ##############################################################
    # Plot only loss
    ##############################################################
    # plot_only_loss(fpath)

    ##############################################################
    # Comparison to Simulations
    ##############################################################
    # compare_to_simulations(fpath)

    ##############################################################
    # Permutations Comparison
    ##############################################################
    # fpath = "./runs/run_2021_04_07__12_44__ds_permutations__N_5/permute_d_5_results.csv"
    # fpath = "./runs/run_2021_04_07__12_44__Ls_permutations__N_5/permute_L_5_results.csv"

    # fpath = "./runs/run_2021_04_24__21_54__initial_strong_ds_permutations__N_5/permute_d_5_results.csv"
    # fpath = "./runs/run_2021_04_24__21_54__initial_strong_Ls_permutations__N_5/permute_L_5_results.csv"
    # fpath = "./runs/run_2021_04_24__22_04__initial_semistrong_end_of_sgd_ds_permutations__N_5/permute_d_5_results.csv"
    # fpath = "./runs/run_2021_04_24__22_04__initial_semistrong_end_of_sgd_Ls_permutations__N_5/permute_L_5_results.csv"

    # is_best_not_worst = True
    # # fpath = "./runs/run_2021_05_24__00_01__perturbed_initial_semistrong_end_of_sgd_ds_permutations__N_5" \
    # #         "/permute_d_5_results.csv"
    # # df = print_sorted_permutations_csv(fpath, is_best_not_worst=is_best_not_worst)
    # # fpath = "./runs/run_2021_05_24__00_00__perturbed_initial_semistrong_end_of_sgd_Ls_permutations__N_5" \
    # #         "/permute_L_5_results.csv"
    # # df = print_sorted_permutations_csv(fpath, is_best_not_worst=is_best_not_worst)
    # fpath = "./runs/run_2021_05_23__23_57__perturbed_initial_strong_ds_permutations__N_5/permute_d_5_results.csv"
    # df = print_sorted_permutations_csv(fpath, is_best_not_worst=is_best_not_worst)
    # df_sorted_lengths = print_sorted_permutations_according_to_lengths(fpath, 'dt',
    # is_best_not_worst=is_best_not_worst)
    # fpath = "./runs/run_2021_05_23__23_57__perturbed_initial_strong_Ls_permutations__N_5/permute_L_5_results.csv"
    # df = print_sorted_permutations_csv(fpath, is_best_not_worst=is_best_not_worst)
    # df_sorted_lengths = print_sorted_permutations_according_to_lengths(fpath, 'Lt',
    # is_best_not_worst=is_best_not_worst)

    ##############################################################
    # Analyze running N's
    ##############################################################
    # df_N = get_all_results_of_path_template("N", "__after_fix_seed_{}__N_{}", value_range=range(1, 15 + 1),
    #                                         seed_range=range(73, 77 + 1))
    # fig, axes = plt.subplots(1, 2)
    # ys = ["p_finish", "reward_conditioned"]
    # ylabels = ["Finish probability", "Time in device (reward)"]
    # titles = ["A", "B"]
    # for i, ax in enumerate(axes):
    #     sns.lineplot(data=df_N, x="N", y=ys[i], ax=ax)
    #     ax.set_xlabel("N")
    #     ax.set_ylabel(ylabels[i])
    #     ax.set_title(titles[i] + ". " + ylabels[i])
    #     ax.grid()
    # axes[0].set_ylim([0, 1])
    # axes[1].set_ylim([0, 10])

    # ##############################################################
    # # Analyze running pf's
    # ##############################################################
    # df_pf = get_all_results_of_path_template("pf", "__after_fix_seed_{}__pf_{:.2f}",
    #                                          value_range=np.arange(0, 1.01, 0.05), seed_range=range(73, 77 + 1))
    # fig, axes = plt.subplots(1, 2)
    # sns.lineplot(data=df_pf, x="pf", y="p_finish", ax=axes[0])
    # axes[0].set_xlabel("Required finish probability")
    # axes[0].set_ylabel("Result finish probability")
    # axes[0].grid()
    #
    # sns.lineplot(data=df_pf, x="pf", y="reward_conditioned", ax=axes[1])
    # axes[1].set_xlabel("Required finish probability")
    # axes[1].set_ylabel("Time in device (reward)")
    # axes[1].set_yscale('log')
    # axes[1].plot(np.arange(0, 1.01, 0.05), [2]*len(np.arange(0, 1.01, 0.05)), 'r--')
    # axes[1].grid()

    # ##############################################################
    # # Analyze running gammas
    # ##############################################################
    df_gamma = get_all_results_of_path_template("gamma", "__after_fix_seed_{}__gamma_{}",
                                                value_range=np.logspace(-1, np.log10(40), 21),
                                                seed_range=range(73, 77 + 1))

    fig, axes = plt.subplots(1, 2)
    ys = ["p_finish", "reward_conditioned"]
    ylabels = ["Finish probability", "Time in device (reward)"]
    titles = ["A", "B"]
    for i, ax in enumerate(axes):
        sns.lineplot(data=df_gamma, x="gamma", y=ys[i], ax=ax)
        ax.set_xscale("log")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i] + ". " + ylabels[i])
        ax.grid()
    axes[0].set_ylim([0, 1.05])
    axes[0].plot(np.logspace(-1, np.log10(40), 21), [0.1] * len(np.logspace(-1, np.log10(40), 21)), 'r--')

    # ##############################################################
    # # Analyze running alphas
    # ##############################################################
    # # df_alpha = get_all_results_of_path_template("alpha", "__new_clamp_seed_{}__alpha_{}",
    # #                                             value_range=np.logspace(-4, -1, 31),
    # #                                             seed_range=range(73, 77 + 1))
    # # plt.figure()
    # # sns.lineplot(data=df_alpha, x="alpha", y="p_scatter")
    # # plt.xscale("log")
    # # plt.xlabel(r"$\alpha$")
    # # plt.ylabel("Scatter probability")
    # # plt.grid()
    # #
    # # plt.figure()
    # # sns.lineplot(data=df_alpha, x="alpha", y="p_finish")
    # # plt.xscale("log")
    # # plt.xlabel(r"$\alpha$")
    # # plt.ylabel("Finish probability")
    # # plt.grid()
    #
    # # plt.figure()
    # # sns.lineplot(data=df_alpha, x="alpha", y="reward_conditioned")
    # # plt.xscale("log")
    # # # plt.yscale("log")
    # # plt.xlabel(r"$\alpha$")
    # # plt.ylabel("Time in device (reward)")
    # # plt.grid()
    # #
    # # plt.figure()
    # # sns.lineplot(data=df_alpha, x="alpha", y="loss")
    # # plt.xscale("log")
    # # # plt.yscale("log")
    # # plt.xlabel(r"$\alpha$")
    # # plt.ylabel("Loss")
    # # plt.grid()
    #
    # # fig, axes = plt.subplots(1, 3)
    # # ys = ["p_finish", "p_scatter", "reward_conditioned"]
    # # ylabels = ["Finish probability", "Scatter probability", "Time in device (reward)"]
    # # titles = ["A", "B", "C"]
    # # for i, ax in enumerate(axes):
    # #     sns.lineplot(data=df_alpha, x="alpha", y=ys[i], ax=ax)
    # #     ax.set_xscale("log")
    # #     ax.set_xlabel(r"$\alpha$")
    # #     ax.set_ylabel(ylabels[i])
    # #     ax.set_title(titles[i] + ". " + ylabels[i])
    # #     ax.grid()
    #
