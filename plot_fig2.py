import sciris as sc
import pandas as pd
import pylab as pl
import numpy as np
import utils as ut
from matplotlib.gridspec import GridSpec


def plot_fig2(
    locations=None, background_scens=None, txvx_ages=None, txvx_efficacies=None
):

    ut.set_font(size=24)
    dfs = sc.autolist()
    econdfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}.obj")
        dfs += df
        econ_df = sc.loadobj(f"{ut.resfolder}/{location}_econ.obj")
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(20)
    width = 0.2

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]

    markers = ["s", "v", "P", "*", "+", "D", "^", "x"]

    fig = pl.figure(constrained_layout=True, figsize=(22, 16))
    spec2 = GridSpec(ncols=3, nrows=2, figure=fig)
    ax1 = fig.add_subplot(spec2[:-1, :])
    ax2 = fig.add_subplot(spec2[-1, 0])
    ax3 = fig.add_subplot(spec2[-1, 1])
    ax4 = fig.add_subplot(spec2[-1, 2])

    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        NoTxV_df = (
            bigdf[
                (bigdf.screen_scen == screen_scen_label)
                & (bigdf.vx_scen == vx_scen_label)
                & (bigdf.txvx_scen == "No TxV")
            ]
            .groupby("year")[
                [
                    "cancers",
                    "cancers_low",
                    "cancers_high",
                    "cancer_deaths",
                    "cancer_deaths_low",
                    "cancer_deaths_high",
                ]
            ]
            .sum()
        )

        NoTxV_econdf = econdf[
            (econdf.screen_scen == screen_scen_label)
            & (econdf.vx_scen == vx_scen_label)
            & (econdf.txvx_scen == "No TxV")
        ][["dalys"]].sum()

        daly_noTxV = NoTxV_econdf["dalys"]

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]

        NoTxV_cancers = np.sum(np.array(NoTxV_df["cancers"])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df["cancers_low"])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df["cancers_high"])[ys:ye])

        years = np.array(NoTxV_df.index)[ys:ye]

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df["cancer_deaths"])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(
            np.array(NoTxV_df["cancer_deaths_low"])[ys:ye]
        )
        NoTxV_cancer_deaths_short_high = np.sum(
            np.array(NoTxV_df["cancer_deaths_high"])[ys:ye]
        )

        for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
            for i_age, txvx_age in enumerate(txvx_ages):
                txvx_scen_label_age = f"Mass TxV, {txvx_efficacy}, age {txvx_age}"
                TxV_df = (
                    bigdf[
                        (bigdf.screen_scen == screen_scen_label)
                        & (bigdf.vx_scen == vx_scen_label)
                        & (bigdf.txvx_scen == txvx_scen_label_age)
                    ]
                    .groupby("year")[
                        [
                            "asr_cancer_incidence",
                            "asr_cancer_incidence_low",
                            "asr_cancer_incidence_high",
                            "cancers",
                            "cancers_low",
                            "cancers_high",
                            "cancer_deaths",
                            "cancer_deaths_low",
                            "cancer_deaths_high",
                        ]
                    ]
                    .sum()
                )

                if ib == 0 and i_age == 0:
                    cancers_averted = (
                        np.array(NoTxV_df["cancers"])[ys:ye]
                        - np.array(TxV_df["cancers"])[ys:ye]
                    )
                    cancers_averted_low = (
                        np.array(NoTxV_df["cancers_low"])[ys:ye]
                        - np.array(TxV_df["cancers_low"])[ys:ye]
                    )
                    cancers_averted_high = (
                        np.array(NoTxV_df["cancers_high"])[ys:ye]
                        - np.array(TxV_df["cancers_high"])[ys:ye]
                    )
                    ax1.plot(
                        years,
                        cancers_averted,
                        color=colors[i_eff],
                        linewidth=3,
                        label=txvx_efficacy,
                    )
                    ax1.fill_between(
                        years,
                        cancers_averted_low,
                        cancers_averted_high,
                        color=colors[i_eff],
                        alpha=0.3,
                    )

                TxV_econdf = econdf[
                    (econdf.screen_scen == screen_scen_label)
                    & (econdf.vx_scen == vx_scen_label)
                    & (econdf.txvx_scen == txvx_scen_label_age)
                ][["dalys"]].sum()

                daly_TxV = TxV_econdf["dalys"]
                dalys_averted = daly_noTxV - daly_TxV

                TxV_cancers = np.sum(np.array(TxV_df["cancers"])[ys:ye])
                TxV_cancers_low = np.sum(np.array(TxV_df["cancers_low"])[ys:ye])
                TxV_cancers_high = np.sum(np.array(TxV_df["cancers_high"])[ys:ye])

                best_TxV_cancer_deaths_short = np.sum(
                    np.array(TxV_df["cancer_deaths"])[ys:ye]
                )
                best_TxV_cancer_deaths_short_high = np.sum(
                    np.array(TxV_df["cancer_deaths_high"])[ys:ye]
                )
                best_TxV_cancer_deaths_short_low = np.sum(
                    np.array(TxV_df["cancer_deaths_low"])[ys:ye]
                )

                averted_cancer_deaths_short = (
                    NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                )
                averted_cancer_deaths_short_high = (
                    NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                )
                averted_cancer_deaths_short_low = (
                    NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low
                )

                TxV_cancers_averted = NoTxV_cancers - TxV_cancers
                TxV_cancers_averted_low = NoTxV_cancers_low - TxV_cancers_low
                TxV_cancers_averted_high = NoTxV_cancers_high - TxV_cancers_high
                if i_eff + ib == 0:
                    ax2.scatter(
                        xes[i_eff][ib],
                        TxV_cancers_averted,
                        marker=markers[i_age],
                        color=colors[i_eff],
                        s=300,
                        label=f"{txvx_age}",
                    )
                else:
                    ax2.scatter(
                        xes[i_eff][ib],
                        TxV_cancers_averted,
                        marker=markers[i_age],
                        color=colors[i_eff],
                        s=300,
                    )
                ax3.scatter(
                    xes[i_eff][ib],
                    averted_cancer_deaths_short,
                    marker=markers[i_age],
                    color=colors[i_eff],
                    s=300,
                )
                ax2.vlines(
                    xes[i_eff][ib],
                    ymin=TxV_cancers_averted_low,
                    color=colors[i_eff],
                    ymax=TxV_cancers_averted_high,
                )
                ax4.scatter(
                    xes[i_eff][ib],
                    dalys_averted,
                    marker=markers[i_age],
                    color=colors[i_eff],
                    s=300,
                )
                ax3.vlines(
                    xes[i_eff][ib],
                    ymin=averted_cancer_deaths_short_low,
                    ymax=averted_cancer_deaths_short_high,
                    color=colors[i_eff],
                )

        ib_labels = background_scens.keys()
        ax2.set_xticks([r + 1.5 * width for r in range(len(r2))], ib_labels)
        ax3.set_xticks([r + 1.5 * width for r in range(len(r2))], ib_labels)
        ax4.set_xticks([r + 1.5 * width for r in range(len(r2))], ib_labels)
        # axes[0].get_xaxis().set_visible(False)
        # axes[1].get_xaxis().set_visible(False)

    ax2.set_ylim(bottom=0)
    ax3.set_ylim(bottom=0)
    ax4.set_ylim(bottom=0)
    sc.SIticks(ax1)
    sc.SIticks(ax2)
    sc.SIticks(ax3)
    sc.SIticks(ax4)

    ax1.set_ylabel(f"Cervical cancer cases averted")
    ax2.set_ylabel(f"Cervical cancer cases averted (2030-2060)")
    ax3.set_ylabel(f"Cervical cancer deaths averted (2030-2060)")
    ax4.set_ylabel(f"DALYs averted (2030-2060)")

    ax2.legend(title="Age of TxV")
    ax1.legend(title="TxV Effectiveness")
    ax2.set_xlabel("Background intervention scenario")
    ax3.set_xlabel("Background intervention scenario")
    ax4.set_xlabel("Background intervention scenario")
    fig.tight_layout()
    fig_name = f"{ut.figfolder}/Fig2.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_fig2_v2(
    locations=None,
    background_scens=None,
    txvx_ages=None,
    txvx_efficacies=None,
    discounting=False,
):

    ut.set_font(size=24)
    dfs = sc.autolist()
    econdfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}.obj")
        dfs += df
        econ_df = sc.loadobj(f"{ut.resfolder}/{location}_econ.obj")
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(20)
    width = 0.2
    standard_le = 88.8

    fig = pl.figure(constrained_layout=True, figsize=(22, 22))
    spec2 = GridSpec(ncols=3, nrows=3, figure=fig)
    ax1 = fig.add_subplot(spec2[0, :])
    ax2 = fig.add_subplot(spec2[1, 0])
    ax3 = fig.add_subplot(spec2[1, 1], sharey=ax2)
    ax4 = fig.add_subplot(spec2[1, 2], sharey=ax2)
    ax5 = fig.add_subplot(spec2[2, 0])
    ax6 = fig.add_subplot(spec2[2, 1], sharey=ax5)
    ax7 = fig.add_subplot(spec2[2, 2], sharey=ax5)
    axes1 = [ax2, ax3, ax4]
    axes2 = [ax5, ax6, ax7]
    all_axes = [axes1, axes2]

    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        NoTxV_df = (
            bigdf[
                (bigdf.screen_scen == screen_scen_label)
                & (bigdf.vx_scen == vx_scen_label)
                & (bigdf.txvx_scen == "No TxV")
            ]
            .groupby("year")[
                [
                    "cancers",
                    "cancers_low",
                    "cancers_high",
                    "cancer_deaths",
                    "cancer_deaths_low",
                    "cancer_deaths_high",
                    "n_tx_vaccinated",
                ]
            ]
            .sum()
        )

        NoTxV_econdf_cancers = (
            econdf[
                (econdf.screen_scen == screen_scen_label)
                & (econdf.vx_scen == vx_scen_label)
                & (econdf.txvx_scen == "No TxV")
            ]
            .groupby("year")[["new_cancers", "new_cancer_deaths"]]
            .sum()
        )

        NoTxV_econdf = (
            econdf[
                (econdf.screen_scen == screen_scen_label)
                & (econdf.vx_scen == vx_scen_label)
                & (econdf.txvx_scen == "No TxV")
            ]
            .groupby("year")[["av_age_cancer_deaths", "av_age_cancers"]]
            .mean()
        )

        if discounting:
            cancers = np.array(
                [
                    i / 1.03**t
                    for t, i in enumerate(NoTxV_econdf_cancers["new_cancers"].values)
                ]
            )
            cancer_deaths = np.array(
                [
                    i / 1.03**t
                    for t, i in enumerate(
                        NoTxV_econdf_cancers["new_cancer_deaths"].values
                    )
                ]
            )
        else:
            cancers = NoTxV_econdf_cancers["new_cancers"].values
            cancer_deaths = NoTxV_econdf_cancers["new_cancer_deaths"].values

        avg_age_ca_death = np.mean(NoTxV_econdf["av_age_cancer_deaths"])
        avg_age_ca = np.mean(NoTxV_econdf["av_age_cancers"])
        ca_years = avg_age_ca_death - avg_age_ca
        yld = np.sum(
            np.sum([0.54 * 0.1, 0.049 * 0.5, 0.451 * 0.3, 0.288 * 0.1])
            * ca_years
            * cancers
        )
        yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
        daly_noTxV = yll + yld

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]

        NoTxV_cancers = np.sum(np.array(NoTxV_df["cancers"])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df["cancers_low"])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df["cancers_high"])[ys:ye])

        years = np.array(NoTxV_df.index)[ys:ye]

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df["cancer_deaths"])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(
            np.array(NoTxV_df["cancer_deaths_low"])[ys:ye]
        )
        NoTxV_cancer_deaths_short_high = np.sum(
            np.array(NoTxV_df["cancer_deaths_high"])[ys:ye]
        )

        for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
            cancers_averted_to_plot = []
            cancer_deaths_averted_to_plot = []
            dalys_averted_to_plot = []
            for i_age, txvx_age in enumerate(txvx_ages):
                txvx_scen_label_age = f"Mass TxV, {txvx_efficacy}, age {txvx_age}"
                TxV_df = (
                    bigdf[
                        (bigdf.screen_scen == screen_scen_label)
                        & (bigdf.vx_scen == vx_scen_label)
                        & (bigdf.txvx_scen == txvx_scen_label_age)
                    ]
                    .groupby("year")[
                        [
                            "asr_cancer_incidence",
                            "asr_cancer_incidence_low",
                            "asr_cancer_incidence_high",
                            "cancers",
                            "cancers_low",
                            "cancers_high",
                            "cancer_deaths",
                            "cancer_deaths_low",
                            "cancer_deaths_high",
                            "n_tx_vaccinated",
                        ]
                    ]
                    .sum()
                )

                if ib == 0 and i_age == 1:
                    cancers_averted = (
                        np.array(NoTxV_df["cancers"])[ys:ye]
                        - np.array(TxV_df["cancers"])[ys:ye]
                    )
                    cancers_averted_low = (
                        np.array(NoTxV_df["cancers_low"])[ys:ye]
                        - np.array(TxV_df["cancers_low"])[ys:ye]
                    )
                    cancers_averted_high = (
                        np.array(NoTxV_df["cancers_high"])[ys:ye]
                        - np.array(TxV_df["cancers_high"])[ys:ye]
                    )
                    ax1.plot(
                        years,
                        cancers_averted,
                        color=colors[i_eff],
                        linewidth=3,
                        label=txvx_efficacy,
                    )
                    ax1.fill_between(
                        years,
                        cancers_averted_low,
                        cancers_averted_high,
                        color=colors[i_eff],
                        alpha=0.3,
                    )

                TxV_econdf_cancers = (
                    econdf[
                        (econdf.screen_scen == screen_scen_label)
                        & (econdf.vx_scen == vx_scen_label)
                        & (econdf.txvx_scen == txvx_scen_label_age)
                    ]
                    .groupby("year")[["new_cancers", "new_cancer_deaths"]]
                    .sum()
                )

                TxV_econdf = (
                    econdf[
                        (econdf.screen_scen == screen_scen_label)
                        & (econdf.vx_scen == vx_scen_label)
                        & (econdf.txvx_scen == txvx_scen_label_age)
                    ]
                    .groupby("year")[["av_age_cancer_deaths", "av_age_cancers"]]
                    .mean()
                )

                if discounting:
                    cancers = np.array(
                        [
                            i / 1.03**t
                            for t, i in enumerate(
                                TxV_econdf_cancers["new_cancers"].values
                            )
                        ]
                    )
                    cancer_deaths = np.array(
                        [
                            i / 1.03**t
                            for t, i in enumerate(
                                TxV_econdf_cancers["new_cancer_deaths"].values
                            )
                        ]
                    )
                else:
                    cancers = TxV_econdf_cancers["new_cancers"].values
                    cancer_deaths = TxV_econdf_cancers["new_cancer_deaths"].values

                avg_age_ca_death = np.mean(TxV_econdf["av_age_cancer_deaths"])
                avg_age_ca = np.mean(TxV_econdf["av_age_cancers"])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(
                    np.sum([0.54 * 0.1, 0.049 * 0.5, 0.451 * 0.3, 0.288 * 0.1])
                    * ca_years
                    * cancers
                )
                yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                daly_TxV = yll + yld
                dalys_averted = daly_noTxV - daly_TxV

                TxV_cancers = np.sum(np.array(TxV_df["cancers"])[ys:ye])
                TxV_cancers_low = np.sum(np.array(TxV_df["cancers_low"])[ys:ye])
                TxV_cancers_high = np.sum(np.array(TxV_df["cancers_high"])[ys:ye])

                best_TxV_cancer_deaths_short = np.sum(
                    np.array(TxV_df["cancer_deaths"])[ys:ye]
                )
                best_TxV_cancer_deaths_short_high = np.sum(
                    np.array(TxV_df["cancer_deaths_high"])[ys:ye]
                )
                best_TxV_cancer_deaths_short_low = np.sum(
                    np.array(TxV_df["cancer_deaths_low"])[ys:ye]
                )

                averted_cancer_deaths_short = (
                    NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                )
                averted_cancer_deaths_short_high = (
                    NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                )
                averted_cancer_deaths_short_low = (
                    NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low
                )

                TxV_cancers_averted = NoTxV_cancers - TxV_cancers
                TxV_cancers_averted_low = NoTxV_cancers_low - TxV_cancers_low
                TxV_cancers_averted_high = NoTxV_cancers_high - TxV_cancers_high
                cancers_averted_to_plot.append(TxV_cancers_averted)
                cancer_deaths_averted_to_plot.append(averted_cancer_deaths_short)
                dalys_averted_to_plot.append(dalys_averted)

            all_axes[0][ib].plot(
                txvx_ages, cancers_averted_to_plot, marker="s", color=colors[i_eff]
            )
            all_axes[1][ib].plot(
                txvx_ages, dalys_averted_to_plot, marker="s", color=colors[i_eff]
            )

        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)
        ax4.get_xaxis().set_visible(False)

    ib_labels = list(background_scens.keys())
    ax2.set_title(ib_labels[0])
    ax3.set_title(ib_labels[1])
    ax4.set_title(ib_labels[2])

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7]:
        ax.set_ylim(bottom=0)
        sc.SIticks(ax)

    ax1.set_ylabel(f"Cervical cancer cases averted")
    ax2.set_ylabel(f"Cervical cancer cases averted\n(2030-2060)")
    ax5.set_ylabel(f"DALYs averted\n(2030-2060)")

    ax1.legend(title="TxV Effectiveness")
    ax5.set_xlabel("Age of TxV")
    ax6.set_xlabel("Age of TxV")
    ax7.set_xlabel("Age of TxV")
    fig.tight_layout()
    fig_name = f"{ut.figfolder}/Fig2_v2.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_fig2_v3(
    locations=None,
    filestem=None,
    background_scens=None,
    txvx_ages=None,
    txvx_efficacies=None,
):

    ut.set_font(size=24)
    dfs = sc.autolist()
    econdfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        dfs += df
        econ_df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(20)
    scen_colors = ["gold", "orange", "red", "darkred"]

    fig = pl.figure(constrained_layout=True, figsize=(22, 22))
    spec2 = GridSpec(ncols=2, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec2[0, :])
    ax2 = fig.add_subplot(spec2[1, 0])
    ax3 = fig.add_subplot(spec2[1, 1], sharey=ax2)
    ax5 = fig.add_subplot(spec2[2, 0])
    ax6 = fig.add_subplot(spec2[2, 1], sharey=ax5)
    ax8 = fig.add_subplot(spec2[3, 0])
    ax9 = fig.add_subplot(spec2[3, 1], sharey=ax8)
    axes1 = [ax2, ax3]
    axes2 = [ax5, ax6]
    axes3 = [ax8, ax9]
    all_axes = [axes1, axes2, axes3]

    width = 0.2  # the width of the bars

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3]

    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        NoTxV_df = (
            bigdf[
                (bigdf.screen_scen == screen_scen_label)
                & (bigdf.vx_scen == vx_scen_label)
                & (bigdf.txvx_scen == "No TxV")
            ]
            .groupby("year")[
                [
                    "cancers",
                    "cancers_low",
                    "cancers_high",
                    "cancer_deaths",
                    "cancer_deaths_low",
                    "cancer_deaths_high",
                ]
            ]
            .sum()
        )

        NoTxV_econdf = econdf[
            (econdf.screen_scen == screen_scen_label)
            & (econdf.vx_scen == vx_scen_label)
            & (econdf.txvx_scen == "No TxV")
        ][["dalys"]].sum()

        daly_noTxV = NoTxV_econdf["dalys"]

        ys = sc.findinds(NoTxV_df.index, 2030)[0]
        ye = sc.findinds(NoTxV_df.index, 2060)[0]

        NoTxV_cancers = np.sum(np.array(NoTxV_df["cancers"])[ys:ye])
        NoTxV_cancers_low = np.sum(np.array(NoTxV_df["cancers_low"])[ys:ye])
        NoTxV_cancers_high = np.sum(np.array(NoTxV_df["cancers_high"])[ys:ye])

        years = np.array(NoTxV_df.index)[ys:ye]

        NoTxV_cancer_deaths_short = np.sum(np.array(NoTxV_df["cancer_deaths"])[ys:ye])
        NoTxV_cancer_deaths_short_low = np.sum(
            np.array(NoTxV_df["cancer_deaths_low"])[ys:ye]
        )
        NoTxV_cancer_deaths_short_high = np.sum(
            np.array(NoTxV_df["cancer_deaths_high"])[ys:ye]
        )

        for i_age, txvx_age in enumerate(txvx_ages):

            for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
                txvx_scen_label_age = f"Mass TxV, {txvx_efficacy}, age {txvx_age}"
                TxV_df = (
                    bigdf[
                        (bigdf.screen_scen == screen_scen_label)
                        & (bigdf.vx_scen == vx_scen_label)
                        & (bigdf.txvx_scen == txvx_scen_label_age)
                    ]
                    .groupby("year")[
                        [
                            "asr_cancer_incidence",
                            "asr_cancer_incidence_low",
                            "asr_cancer_incidence_high",
                            "cancers",
                            "cancers_low",
                            "cancers_high",
                            "cancer_deaths",
                            "cancer_deaths_low",
                            "cancer_deaths_high",
                            "n_tx_vaccinated",
                        ]
                    ]
                    .sum()
                )

                if ib == 0 and i_age == 1:
                    cancers_averted = (
                        np.array(NoTxV_df["cancers"])[ys:ye]
                        - np.array(TxV_df["cancers"])[ys:ye]
                    )
                    cancers_averted_low = (
                        np.array(NoTxV_df["cancers_low"])[ys:ye]
                        - np.array(TxV_df["cancers_low"])[ys:ye]
                    )
                    cancers_averted_high = (
                        np.array(NoTxV_df["cancers_high"])[ys:ye]
                        - np.array(TxV_df["cancers_high"])[ys:ye]
                    )
                    ax1.plot(
                        years,
                        cancers_averted,
                        color=colors[i_eff],
                        linewidth=3,
                        label=txvx_efficacy,
                    )
                    ax1.fill_between(
                        years,
                        cancers_averted_low,
                        cancers_averted_high,
                        color=colors[i_eff],
                        alpha=0.3,
                    )

                TxV_econdf = econdf[
                    (econdf.screen_scen == screen_scen_label)
                    & (econdf.vx_scen == vx_scen_label)
                    & (econdf.txvx_scen == txvx_scen_label_age)
                ][["dalys"]].sum()

                daly_TxV = TxV_econdf["dalys"]
                dalys_averted = daly_noTxV - daly_TxV

                TxV_cancers = np.sum(np.array(TxV_df["cancers"])[ys:ye])
                TxV_cancers_low = np.sum(np.array(TxV_df["cancers_low"])[ys:ye])
                TxV_cancers_high = np.sum(np.array(TxV_df["cancers_high"])[ys:ye])

                best_TxV_cancer_deaths_short = np.sum(
                    np.array(TxV_df["cancer_deaths"])[ys:ye]
                )
                best_TxV_cancer_deaths_short_high = np.sum(
                    np.array(TxV_df["cancer_deaths_high"])[ys:ye]
                )
                best_TxV_cancer_deaths_short_low = np.sum(
                    np.array(TxV_df["cancer_deaths_low"])[ys:ye]
                )

                averted_cancer_deaths_short = (
                    NoTxV_cancer_deaths_short - best_TxV_cancer_deaths_short
                )
                averted_cancer_deaths_short_high = (
                    NoTxV_cancer_deaths_short_high - best_TxV_cancer_deaths_short_high
                )
                averted_cancer_deaths_short_low = (
                    NoTxV_cancer_deaths_short_low - best_TxV_cancer_deaths_short_low
                )

                TxV_cancers_averted = NoTxV_cancers - TxV_cancers
                TxV_cancers_averted_low = NoTxV_cancers_low - TxV_cancers_low
                TxV_cancers_averted_high = NoTxV_cancers_high - TxV_cancers_high

                if i_eff + ib == 0:
                    all_axes[0][i_eff].bar(
                        xes[i_age][ib],
                        TxV_cancers_averted,
                        width,
                        color=colors[i_age + 4],
                        label=txvx_age,
                    )
                else:
                    all_axes[0][i_eff].bar(
                        xes[i_age][ib],
                        TxV_cancers_averted,
                        width,
                        color=colors[i_age + 4],
                    )
                all_axes[1][i_eff].bar(
                    xes[i_age][ib],
                    averted_cancer_deaths_short,
                    width,
                    color=colors[i_age + 4],
                )
                all_axes[2][i_eff].bar(
                    xes[i_age][ib],
                    dalys_averted,
                    width,
                    color=colors[i_age + 4],
                )

        ax2.get_xaxis().set_visible(False)
        ax3.get_xaxis().set_visible(False)

        ax5.get_xaxis().set_visible(False)
        ax6.get_xaxis().set_visible(False)

    ax2.set_title(txvx_efficacies[0])
    ax3.set_title(txvx_efficacies[1])

    for ax in [ax1, ax2, ax3, ax5, ax6, ax8, ax9]:
        ax.set_ylim(bottom=0)
        sc.SIticks(ax)

    ax8.set_xticks(r1 + 1 * width, background_scens.keys())
    ax9.set_xticks(r1 + 1 * width, background_scens.keys())

    ax1.set_ylabel(f"Cervical cancer cases averted")
    ax2.set_ylabel(f"Cervical cancer cases averted\n(2030-2060)")
    ax5.set_ylabel(f"Cervical cancer deaths averted\n(2030-2060)")
    ax8.set_ylabel(f"DALYs averted\n(2030-2060)")

    ax1.legend(title="TxV Effectiveness")
    ax2.legend(title="Age of TxV")
    ax8.set_xlabel("TxV Effectiveness")
    ax9.set_xlabel("TxV Effectiveness")
    fig.tight_layout()
    fig_name = f"{ut.figfolder}/Fig2_v3.png"
    fig.savefig(fig_name, dpi=100)

    return


# %% Run as a script
if __name__ == "__main__":
    T = sc.timer()

    locations = [
        "india",  # 0
        "indonesia",  # 1
        "nigeria",  # 2
        "tanzania",  # 3
        "bangladesh",  # 4
        "myanmar",  # 5
        "uganda",  # 6
        "ethiopia",  # 7
        "drc",  # 8
        # 'kenya'  # 9
    ]

    plot_fig2_v3(
        locations=locations,
        background_scens={
            "Scenario 1": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "No screening",
            },
            "Scenario 2": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 35% sc cov, 50% LTFU",
            },
            "Scenario 3": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 70% sc cov, 50% LTFU",
            },
            "Scenario 4": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 70% sc cov, 30% LTFU",
            },
        },
        txvx_efficacies=["70/30", "90/50"],
        txvx_ages=["30", "35", "40"],
        filestem="_feb25",
    )

    T.toc("Done")
