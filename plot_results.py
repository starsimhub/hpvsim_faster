import sciris as sc
import pandas as pd
import pylab as pl
import numpy as np
import utils as ut
import analyzers as an
import seaborn as sns
from matplotlib import cm

from matplotlib.gridspec import GridSpec

resfolder = "results"
figfolder = "figures"
datafolder = "data"

cost_dict = dict(poc_hpv=7, hpv=15, txv=6.92, leep=41.76, ablation=11.76, cancer=450)


def plot_burden_redux(locations, background_scen):
    ut.set_font(size=24)
    dfs = sc.autolist()
    dfs_2030 = sc.autolist()
    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}_feb25.obj")
        dfs += df

        df_2030 = sc.loadobj(f"{ut.resfolder}/{location}_feb23.obj")
        dfs_2030 += df_2030
    bigdf = pd.concat(dfs)

    bigdf_2030 = pd.concat(dfs_2030)
    colormap = cm.viridis
    colors = [colormap(0.9), colormap(0.5), colormap(0.2), colormap(0.72), "red"]

    fig, axes = pl.subplots(ncols=2, figsize=(20, 8))
    vx_scen_label = background_scen[0]
    screen_scen_label = background_scen[1]

    txvx_base_label = "Mass TxV, 90/50, age 30"

    # First No TxV
    df = (
        bigdf_2030[
            (bigdf_2030.screen_scen == screen_scen_label)
            & (bigdf_2030.vx_scen == vx_scen_label)
            & (bigdf_2030.txvx_scen == "No TxV")
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2020:]
    )
    years = np.array(df.index)
    axes[0].plot(years, df["cancers"], color=colors[0], label="No TxV")
    axes[0].fill_between(
        years, df["cancers_low"], df["cancers_high"], color=colors[0], alpha=0.3
    )
    df = (
        bigdf_2030[
            (bigdf_2030.screen_scen == screen_scen_label)
            & (bigdf_2030.vx_scen == vx_scen_label)
            & (bigdf_2030.txvx_scen == "No TxV")
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2030:]
    )
    cum_cases = np.sum(df["cancers"])
    cum_cases_2040 = np.sum(df["cancers"].values[:10])
    axes[1].bar(0, cum_cases, color=colors[0], label="2030-2060")
    axes[1].bar(0, cum_cases_2040, color=colors[0], hatch="///", label="2030-2040")
    axes[1].text(
        0,
        cum_cases_2040 + 10e4,
        round(cum_cases_2040 / 1e6, 2),
        ha="center",
    )

    axes[1].text(
        0,
        cum_cases + 10e4,
        round(cum_cases / 1e6, 2),
        ha="center",
    )
    # Now POC HPV TxV
    df = (
        bigdf_2030[
            (bigdf_2030.screen_scen == "HPV, 35% sc cov, 30% LTFU")
            & (bigdf_2030.vx_scen == vx_scen_label)
            & (bigdf_2030.txvx_scen == "No TxV")
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2020:]
    )
    years = np.array(df.index)
    axes[0].plot(years, df["cancers"], color=colors[4], label="POC HPV Test")
    axes[0].fill_between(
        years, df["cancers_low"], df["cancers_high"], color=colors[4], alpha=0.3
    )
    df = (
        bigdf_2030[
            (bigdf_2030.screen_scen == "HPV, 35% sc cov, 30% LTFU")
            & (bigdf_2030.vx_scen == vx_scen_label)
            & (bigdf_2030.txvx_scen == "No TxV")
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2030:]
    )
    cum_cases = np.sum(df["cancers"])
    cum_cases_2040 = np.sum(df["cancers"].values[:10])
    axes[1].bar(1, cum_cases, color=colors[4])
    axes[1].bar(1, cum_cases_2040, color=colors[4], hatch="///")
    axes[1].text(
        1,
        cum_cases_2040 + 10e4,
        round(cum_cases_2040 / 1e6, 2),
        ha="center",
    )
    axes[1].text(
        1,
        cum_cases + 10e4,
        round(cum_cases / 1e6, 2),
        ha="center",
    )

    # First let's do 2030
    df = (
        bigdf_2030[
            (bigdf_2030.screen_scen == screen_scen_label)
            & (bigdf_2030.vx_scen == vx_scen_label)
            & (bigdf_2030.txvx_scen == txvx_base_label)
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2020:]
    )
    years = np.array(df.index)
    axes[0].plot(years, df["cancers"], color=colors[1], label="Mass TxV in 2030")
    axes[0].fill_between(
        years, df["cancers_low"], df["cancers_high"], color=colors[1], alpha=0.3
    )
    df = (
        bigdf_2030[
            (bigdf_2030.screen_scen == screen_scen_label)
            & (bigdf_2030.vx_scen == vx_scen_label)
            & (bigdf_2030.txvx_scen == txvx_base_label)
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2030:]
    )
    cum_cases = np.sum(df["cancers"])
    cum_cases_2040 = np.sum(df["cancers"].values[:10])
    axes[1].bar(2, cum_cases, color=colors[1])
    axes[1].bar(2, cum_cases_2040, color=colors[1], hatch="///")
    axes[1].text(
        2,
        cum_cases_2040 + 10e4,
        round(cum_cases_2040 / 1e6, 2),
        ha="center",
    )
    axes[1].text(
        2,
        cum_cases + 10e4,
        round(cum_cases / 1e6, 2),
        ha="center",
    )

    # Now 2035

    df = (
        bigdf[
            (bigdf.screen_scen == screen_scen_label)
            & (bigdf.vx_scen == vx_scen_label)
            & (bigdf.txvx_scen == txvx_base_label)
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2020:]
    )
    years = np.array(df.index)
    axes[0].plot(years, df["cancers"], color=colors[2], label="Mass TxV in 2035")
    axes[0].fill_between(
        years, df["cancers_low"], df["cancers_high"], color=colors[2], alpha=0.3
    )

    df = (
        bigdf[
            (bigdf.screen_scen == screen_scen_label)
            & (bigdf.vx_scen == vx_scen_label)
            & (bigdf.txvx_scen == txvx_base_label)
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2030:]
    )
    cum_cases = np.sum(df["cancers"])
    cum_cases_2040 = np.sum(df["cancers"].values[:10])
    axes[1].bar(3, cum_cases, color=colors[2])
    axes[1].bar(3, cum_cases_2040, color=colors[2], hatch="///")
    axes[1].text(
        3,
        cum_cases_2040 + 10e4,
        round(cum_cases_2040 / 1e6, 2),
        ha="center",
    )
    axes[1].text(
        3,
        cum_cases + 10e4,
        round(cum_cases / 1e6, 2),
        ha="center",
    )
    # Now 2038
    df = (
        bigdf[
            (bigdf.screen_scen == screen_scen_label)
            & (bigdf.vx_scen == vx_scen_label)
            & (bigdf.txvx_scen == f"{txvx_base_label}, intro 2038")
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2020:]
    )
    years = np.array(df.index)
    axes[0].plot(years, df["cancers"], color=colors[3], label="Mass TxV in 2038")
    axes[0].fill_between(
        years, df["cancers_low"], df["cancers_high"], color=colors[3], alpha=0.3
    )
    df = (
        bigdf[
            (bigdf.screen_scen == screen_scen_label)
            & (bigdf.vx_scen == vx_scen_label)
            & (bigdf.txvx_scen == f"{txvx_base_label}, intro 2038")
        ]
        .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
        .sum()[2030:]
    )
    cum_cases = np.sum(df["cancers"])
    cum_cases_2040 = np.sum(df["cancers"].values[:10])
    axes[1].bar(4, cum_cases, color=colors[3])
    axes[1].bar(4, cum_cases_2040, color=colors[3], hatch="///")
    axes[1].text(
        4,
        cum_cases_2040 + 10e4,
        round(cum_cases_2040 / 1e6, 2),
        ha="center",
    )
    axes[1].text(
        4,
        cum_cases + 10e4,
        round(cum_cases / 1e6, 2),
        ha="center",
    )
    axes[0].set_ylabel("Cervical cancer cases ")
    axes[1].set_ylabel("Cervical cancer cases")
    axes[1].set_xticks([0, 1, 2, 3, 4], ["No TxV", "POC HPV", "2030", "2035", "2038"])
    for ax in axes:
        sc.SIticks(ax)
    axes[0].legend()
    axes[1].legend()
    axes[1].set_ylim(top=14.5e6)
    fig.tight_layout()
    fig_name = f"{figfolder}/txv_burden_redux_ts.png"
    sc.savefig(fig_name, dpi=100)
    return


def plot_residual_burden_combined(locations, scens, filestem):

    ut.set_font(size=24)
    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        dfs += df
    bigdf = pd.concat(dfs)
    colors = ["gold", "orange", "red", "darkred"]

    fig = pl.figure(constrained_layout=True, figsize=(16, 20))
    spec2 = GridSpec(ncols=3, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec2[0, :])
    axes = []
    for i in np.arange(1, 4):
        for j in np.arange(0, 3):
            ax = fig.add_subplot(spec2[i, j])
            axes.append(ax)
    txvx_scen_label = "No TxV"
    for ib, (ib_label, ib_scens) in enumerate(scens.items()):
        vx_scen_label = ib_scens[0]
        screen_scen_label = ib_scens[1]
        df = (
            bigdf[
                (bigdf.screen_scen == screen_scen_label)
                & (bigdf.vx_scen == vx_scen_label)
                & (bigdf.txvx_scen == txvx_scen_label)
            ]
            .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
            .sum()[2020:]
        )
        years = np.array(df.index)
        ax1.plot(years, df["cancers"], color=colors[ib], label=ib_label)
        ax1.fill_between(
            years, df["cancers_low"], df["cancers_high"], color=colors[ib], alpha=0.3
        )
        ax1.set_title("9 countries combined")
        ax1.set_ylabel("Cervical cancer cases")

    for pn, location in enumerate(locations):
        ax = axes[pn]
        for ib, (ib_label, ib_scens) in enumerate(scens.items()):
            vx_scen_label = ib_scens[0]
            screen_scen_label = ib_scens[1]
            df = (
                bigdf[
                    (bigdf.screen_scen == screen_scen_label)
                    & (bigdf.vx_scen == vx_scen_label)
                    & (bigdf.txvx_scen == txvx_scen_label)
                    & (bigdf.location == location)
                ]
                .groupby("year")[["cancers", "cancers_low", "cancers_high"]]
                .sum()[2020:]
            )
            years = np.array(df.index)
            ax.plot(years, df["cancers"], color=colors[ib])
            ax.fill_between(
                years,
                df["cancers_low"],
                df["cancers_high"],
                color=colors[ib],
                alpha=0.3,
            )
            sc.SIticks(ax)
            if location == "drc":
                ax.set_title("DRC")
            else:
                ax.set_title(location.capitalize())
    for pn in range(len(locations)):
        axes[pn].set_ylim(bottom=0)
    sc.SIticks(ax1)
    ax1.legend()
    fig.tight_layout()
    fig_name = f"{figfolder}/residual_burden.png"
    sc.savefig(fig_name, dpi=100)

    return


def plot_fig1(locations, scens, txv_scens, filestem):

    ut.set_font(size=24)
    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        dfs += df

    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(10)
    x = np.arange(len(scens))  # the label locations
    width = 0.2  # the width of the bars

    r1 = np.arange(len(scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]

    xes = [r1, r2, r3, r4]
    fig, ax = pl.subplots(figsize=(16, 8))
    for ib, (ib_label, ib_scens) in enumerate(scens.items()):
        for it, (txvx_scen, txvx_scen_label) in enumerate(txv_scens.items()):
            vx_scen_label = ib_scens[0]
            screen_scen_label = ib_scens[1]
            df = (
                bigdf[
                    (bigdf.screen_scen == screen_scen_label)
                    & (bigdf.vx_scen == vx_scen_label)
                    & (bigdf.txvx_scen == txvx_scen_label)
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
                .sum()[2030:]
            )

            years = np.array(df.index)
            cum_cases = np.sum(df["cancers"])

            if txvx_scen_label == "No TxV":
                txvx_scen_label_to_plot = txvx_scen_label
            else:
                txvx_scen_label_to_plot = txvx_scen
            if ib == 0:
                ax.bar(
                    xes[it][ib],
                    cum_cases,
                    width,
                    color=colors[it],
                    edgecolor="black",
                    label=txvx_scen_label_to_plot,
                )
            else:
                ax.bar(
                    xes[it][ib], cum_cases, width, color=colors[it], edgecolor="black"
                )

            ax.text(
                xes[it][ib], cum_cases + 10e4, round(cum_cases / 1e6, 1), ha="center"
            )

    ax.set_ylabel("Cervical cancer cases (2030-2060)")
    ax.set_xticks(x + 1.5 * width, scens.keys())
    ax.set_xlabel("Background intervention scenario (PxV-Sc-Tx)")
    ax.set_ylim(top=14.5e6)
    sc.SIticks(ax)
    ax.legend(ncol=2)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CC_burden{filestem}.png"
    sc.savefig(fig_name, dpi=100)

    return


def plot_fig1_v2(locations, scens, filestem):

    ut.set_font(size=24)
    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        dfs += df

    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(10)
    x = np.arange(len(scens))  # the label locations
    width = 0.2  # the width of the bars

    fig, axes = pl.subplots(nrows=2, figsize=(10, 10))
    for ib, ib_label in enumerate(scens):
        df = (bigdf[(bigdf.scenario == ib_label)]
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
                    .sum()[2025:]
                )

        cum_cases = np.sum(df["cancers"])
        axes[0].plot(df.index, df["cancers"], color=colors[ib], label=ib_label)
        axes[0].fill_between(
            df.index,
            df["cancers_low"],
            df["cancers_high"],
            color=colors[ib],
            alpha=0.3,
        )
        axes[1].bar(
                        x[ib],
                        cum_cases,
                        color=colors[ib],
                        edgecolor="black",
                    )
        axes[1].text(x[ib],
                cum_cases+ 10e4,
                round(cum_cases / 1e6, 1),
                ha="center",
                )
               
    axes[1].set_xticks(x, scens)
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylabel("Cervical cancer cases")
    axes[0].set_xlabel("Year")
    axes[1].set_ylabel("Cervical cancer cases (2025-2060)")
    axes[1].set_xlabel("Scenario")
    axes[0].legend()
    for ax in axes:
        sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CC_burden_{filestem}.png"
    sc.savefig(fig_name, dpi=100)

    return


def plot_fig2(
    locations=None,
    background_scen=None,
    txvx_efficacy=None,
    txvx_ages=None,
    sensitivities=None,
    filestem=None,
):

    sens_labels = {
        "": ["Baseline", "baseline"],
        ", cross-protection": ["50% cross-protection", "50%\ncross-protection"],
        ", 0.05 decay": ["5% annual decay since virus/lesion", "0.05decay"],
        ", intro 2038": ["2038 introduction", "2038\nintroduction"],
        ", no durable immunity": ["No immune memory", "No immune\nmemory"],
        "70/30": [
            "70% HPV clearance,\n30% CIN2+ clearance",
            "70% HPV\nclearance,\n30% CIN2+\nclearance",
        ],
        ", 50 cov": ["50% coverage", "50% \ncoverage"],
    }
    sens_labels_to_use = [sens_labels[sens_label][0] for sens_label in sensitivities]
    ut.set_font(size=20)
    econdfs = sc.autolist()

    for location in locations:
        econ_df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        econdfs += econ_df

    econdf = pd.concat(econdfs)
    colors = sc.gridcolors(20)[4:]
    width = 0.2

    r1 = np.arange(len(sensitivities))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    fig, ax = pl.subplots(figsize=(14, 8))
    vx_scen_label = background_scen["vx_scen"]
    screen_scen_label = background_scen["screen_scen"]
    NoTxV = (
        econdf[
            (econdf.screen_scen == screen_scen_label)
            & (econdf.vx_scen == vx_scen_label)
            & (econdf.txvx_scen == "No TxV")
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

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV["cancers"])[ys:ye])

    for i_age, txvx_age in enumerate(txvx_ages):
        txvx_scen_label = f"Mass TxV, {txvx_efficacy}, age {txvx_age}"

        TxV_baseline = (
            econdf[
                (econdf.screen_scen == screen_scen_label)
                & (econdf.vx_scen == vx_scen_label)
                & (econdf.txvx_scen == txvx_scen_label)
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

        TxV_cancers_baseline = np.sum(np.array(TxV_baseline["cancers"])[ys:ye])
        if i_age == 0:
            ax.axhline(
                NoTxV_cancers - TxV_cancers_baseline,
                color=colors[i_age],
                linewidth=2,
                label="Baseline TxV",
            )
        else:
            ax.axhline(
                NoTxV_cancers - TxV_cancers_baseline, color=colors[i_age], linewidth=2
            )

        for isens, sens_label in enumerate(sensitivities):
            if sens_label == "70/30":
                txvx_scen_label_sen = f"Mass TxV, 70/30, age {txvx_age}"
            else:
                txvx_scen_label_sen = f"{txvx_scen_label}{sens_label}"

            TxV = (
                econdf[
                    (econdf.screen_scen == screen_scen_label)
                    & (econdf.vx_scen == vx_scen_label)
                    & (econdf.txvx_scen == txvx_scen_label_sen)
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
            if len(TxV):
                ys = sc.findinds(TxV.index, 2030)[0]
                ye = sc.findinds(TxV.index, 2060)[0]
                TxV_cancers = np.sum(np.array(TxV["cancers"])[ys:ye])
                averted_cancers = NoTxV_cancers - TxV_cancers

                if isens == 0:
                    ax.bar(
                        xes[i_age][isens],
                        averted_cancers,
                        color=colors[i_age],
                        width=width,
                        edgecolor="black",
                        label=txvx_age,
                    )
                    # ax.scatter(xes[i_age][isens], averted_cancers,
                    #            color=colors[i_age], s=400, edgecolor='black', label=txvx_age)

                else:
                    ax.bar(
                        xes[i_age][isens],
                        averted_cancers,
                        width=width,
                        edgecolor="black",
                        color=colors[i_age],
                    )
                    # ax.scatter(xes[i_age][isens], averted_cancers, s=400, edgecolor='black',
                    #            color=colors[i_age])

    sc.SIticks(ax)
    # ax.set_ylim(9e5,3e6)
    ax.set_ylim(0, 3e6)
    ax.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use)

    ax.set_ylabel(f"Cervical cancer cases averted")

    ax.legend(title="Age of TxV", ncol=2)
    fig.tight_layout()

    fig_name = f"{ut.figfolder}/sensitivity{filestem}.png"
    fig.savefig(fig_name, dpi=100)
    return


def plot_fig2_PDVAC(
    locations=None,
    background_scen=None,
    txvx_efficacy=None,
    txvx_age=None,
    sensitivities=None,
    filestem=None,
):

    sens_labels = {
        "": ["Baseline", "baseline"],
        ", cross-protection": ["50% cross-protection", "50%\ncross-protection"],
        ", 0.05 decay": ["5% annual decay since virus/lesion", "0.05decay"],
        ", intro 2035": ["2035 introduction", "2035\nintroduction"],
        ", no durable immunity": ["No immune memory", "No immune\nmemory"],
        "70/30": [
            "70% HPV clearance,\n30% CIN2+ clearance",
            "70% HPV\nclearance,\n30% CIN2+\nclearance",
        ],
        ", 50 cov": ["50% coverage", "50% \ncoverage"],
        "TnV TxV, 90/50": ["Test and vaccinate", "Test and\nvaccinate"],
    }
    sens_labels_to_use = [sens_labels[sens_label][0] for sens_label in sensitivities]
    ut.set_font(size=20)
    econdfs = sc.autolist()

    delayed_intro_dfs = sc.autolist()

    for location in locations:
        econ_df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        econdfs += econ_df
        delayed_intro_df = sc.loadobj(f"{ut.resfolder}/{location}_delayed_intro.obj")
        delayed_intro_dfs += delayed_intro_df

    econdf = pd.concat(econdfs)
    delayed_df = pd.concat(delayed_intro_dfs)
    colors = sc.gridcolors(20)[4:]
    width = 0.4

    xes = np.arange(len(sensitivities))
    fig, ax = pl.subplots(figsize=(14, 8))
    vx_scen_label = background_scen["vx_scen"]
    screen_scen_label = background_scen["screen_scen"]
    NoTxV = (
        econdf[
            (econdf.screen_scen == screen_scen_label)
            & (econdf.vx_scen == vx_scen_label)
            & (econdf.txvx_scen == "No TxV")
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

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV["cancers"])[ys:ye])
    txvx_scen_label = f"Mass TxV, {txvx_efficacy}, age {txvx_age}"

    TxV_baseline = (
        econdf[
            (econdf.screen_scen == screen_scen_label)
            & (econdf.vx_scen == vx_scen_label)
            & (econdf.txvx_scen == txvx_scen_label)
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

    TxV_cancers_baseline = np.sum(np.array(TxV_baseline["cancers"])[ys:ye])
    ax.axhline(NoTxV_cancers - TxV_cancers_baseline, linewidth=2, label="Baseline TxV")

    for isens, sens_label in enumerate(sensitivities):
        if sens_label == "70/30":
            txvx_scen_label_sen = f"Mass TxV, 70/30, age {txvx_age}"
        elif "TnV" in sens_label:
            txvx_scen_label_sen = f"{sens_label}, age {txvx_age}"
        else:
            txvx_scen_label_sen = f"{txvx_scen_label}{sens_label}"

        if "intro 2035" in sens_label:
            TxV = (
                delayed_df[
                    (delayed_df.screen_scen == screen_scen_label)
                    & (delayed_df.vx_scen == vx_scen_label)
                    & (delayed_df.txvx_scen == txvx_scen_label_sen)
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
            NoTxV_delayed = (
                delayed_df[
                    (delayed_df.screen_scen == screen_scen_label)
                    & (delayed_df.vx_scen == vx_scen_label)
                    & (delayed_df.txvx_scen == "No TxV")
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

            ys = sc.findinds(TxV.index, 2035)[0]
            ye = sc.findinds(TxV.index, 2065)[0]
            TxV_cancers = np.sum(np.array(TxV["cancers"])[ys:ye])
            no_TxV_cancers = np.sum(np.array(NoTxV_delayed["cancers"])[ys:ye])
            averted_cancers = no_TxV_cancers - TxV_cancers
        else:
            TxV = (
                econdf[
                    (econdf.screen_scen == screen_scen_label)
                    & (econdf.vx_scen == vx_scen_label)
                    & (econdf.txvx_scen == txvx_scen_label_sen)
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
            ys = sc.findinds(TxV.index, 2030)[0]
            ye = sc.findinds(TxV.index, 2060)[0]
            TxV_cancers = np.sum(np.array(TxV["cancers"])[ys:ye])
            averted_cancers = NoTxV_cancers - TxV_cancers

        ax.bar(
            xes[isens], averted_cancers, width=width, color=colors[0], edgecolor="black"
        )
        # ax.scatter(xes[i_age][isens], averted_cancers, s=400, edgecolor='black',
        #            color=colors[i_age])

    sc.SIticks(ax)
    # ax.set_ylim(9e5,3e6)
    ax.set_ylim(0, 3e6)
    ax.set_xticks(xes, sens_labels_to_use)

    ax.set_ylabel(f"Cervical cancer cases averted")

    ax.legend()
    fig.tight_layout()

    fig_name = f"{ut.figfolder}/sensitivity_PDVAC.png"
    fig.savefig(fig_name, dpi=100)
    return


def plot_tnv_sens(
    locations=None,
    background_scen=None,
    txvx_efficacies=None,
    txvx_age=None,
    filestem=None,
):

    ut.set_font(size=20)
    dfs = sc.autolist()

    for location in locations:
        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        dfs += df

    df = pd.concat(dfs)
    colors = sc.gridcolors(20)[7:]
    width = 0.25

    r1 = np.arange(len(txvx_efficacies))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    fig, ax = pl.subplots(figsize=(14, 8))
    vx_scen_label = background_scen["vx_scen"]
    screen_scen_label = background_scen["screen_scen"]
    NoTxV = (
        df[
            (df.screen_scen == screen_scen_label)
            & (df.vx_scen == vx_scen_label)
            & (df.txvx_scen == "No TxV")
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

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV["cancers"])[ys:ye])

    for itxv, txvx_efficacy in enumerate(txvx_efficacies):

        txvx_scen_label = f"TnV TxV, {txvx_efficacy}, age {txvx_age}"
        TxV = (
            df[
                (df.screen_scen == screen_scen_label)
                & (df.vx_scen == vx_scen_label)
                & (df.txvx_scen == txvx_scen_label)
            ]
            .groupby("year")[
                [
                    "cancers",
                    "cancers_low",
                    "cancers_high",
                    "cancer_deaths",
                    "cancer_deaths_low",
                    "cancer_deaths_high",
                    "new_txvx_doses",
                    "new_screens",
                ]
            ]
            .sum()
        )
        ys = sc.findinds(TxV.index, 2030)[0]
        ye = sc.findinds(TxV.index, 2060)[0]
        TxV_cancers = np.sum(np.array(TxV["cancers"])[ys:ye])
        averted_cancers = NoTxV_cancers - TxV_cancers
        TxV_vaccinations = np.sum(np.array(TxV["new_txvx_doses"])[ys:ye])

        mass_txvx_scen_label = f"Mass TxV, {txvx_efficacy}, age {txvx_age}"
        mass_TxV = (
            df[
                (df.screen_scen == screen_scen_label)
                & (df.vx_scen == vx_scen_label)
                & (df.txvx_scen == mass_txvx_scen_label)
            ]
            .groupby("year")[
                [
                    "cancers",
                    "cancers_low",
                    "cancers_high",
                    "cancer_deaths",
                    "cancer_deaths_low",
                    "cancer_deaths_high",
                    "new_txvx_doses",
                    "new_screens",
                ]
            ]
            .sum()
        )
        ys = sc.findinds(TxV.index, 2030)[0]
        ye = sc.findinds(TxV.index, 2060)[0]
        mass_TxV_cancers = np.sum(np.array(mass_TxV["cancers"])[ys:ye])
        mass_averted_cancers = NoTxV_cancers - mass_TxV_cancers

        mass_TxV_vaccinations = np.sum(np.array(mass_TxV["new_txvx_doses"])[ys:ye])

        if itxv == 0:
            ax.bar(
                xes[0][itxv],
                averted_cancers,
                color=colors[0],
                width=width,
                edgecolor="black",
                label="Test and vaccinate",
            )
            ax.bar(
                xes[1][itxv],
                mass_averted_cancers,
                color=colors[1],
                width=width,
                edgecolor="black",
                label="Mass vaccinate",
            )

        else:
            ax.bar(
                xes[0][itxv],
                averted_cancers,
                width=width,
                edgecolor="black",
                color=colors[0],
            )
            ax.bar(
                xes[1][itxv],
                mass_averted_cancers,
                color=colors[1],
                width=width,
                edgecolor="black",
            )

        ax.text(
            xes[0][itxv],
            averted_cancers + 2e4,
            round(averted_cancers / 1e6, 1),
            ha="center",
        )
        ax.text(
            xes[1][itxv],
            mass_averted_cancers + 2e4,
            round(mass_averted_cancers / 1e6, 1),
            ha="center",
        )

    print(
        f"{round(TxV_vaccinations/1e6,0)} million TxV (TnV), {round(mass_TxV_vaccinations/1e6,0)} million TxV (Mass Vx)"
    )
    sc.SIticks(ax)
    ax.set_xticks([r + width / 2 for r in range(len(r1))], txvx_efficacies)

    ax.set_ylabel(f"Cervical cancer cases averted")
    ax.set_xlabel(f"TxV % Effectiveness (HPV/CIN2+ clearance)")

    ax.legend(title="TxV Delivery (age 30-40)")
    fig.tight_layout()

    fig_name = f"{ut.figfolder}/tnv_sens{filestem}.png"
    fig.savefig(fig_name, dpi=100)


def plot_CEA_sensv2(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    filestem=None,
):
    ut.set_font(size=14)
    econdfs = sc.autolist()
    fasterecondfs = sc.autolist()

    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        econdfs += econdf

        faster_econdf = sc.loadobj(f"{ut.resfolder}/{location}_feb24_econ.obj")
        fasterecondfs += faster_econdf

    econ_df = pd.concat(econdfs)
    faster_econ_df = pd.concat(fasterecondfs)

    handles = []
    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "D", "h", "X", "o", "d"]
    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            if "HPV" in txvx_scen and txvx_scen[:-10] != screen_scen_label[:-10]:
                pass
            else:
                dalys_noTxV = 0
                dalys_TxV = 0
                costs_noTxV = 0
                costs_TxV = 0
                for location in locations:
                    econ_df_to_use = econ_df
                    NoTxV = sc.dcp(
                        econ_df_to_use[
                            (econ_df_to_use.screen_scen == screen_scen_label)
                            & (econ_df_to_use.vx_scen == vx_scen_label)
                            & (econ_df_to_use.txvx_scen == "No TxV")
                            & (econ_df_to_use.location == location)
                        ][
                            [
                                "dalys",
                                "new_poc_hpv_screens",
                                "new_hpv_screens",
                                "new_vaccinations",
                                "new_tx_vaccinations",
                                "new_thermal_ablations",
                                "new_leeps",
                                "new_cancer_treatments",
                            ]
                        ]
                    )

                    dalys_noTxV += NoTxV["dalys"]

                    cost_noTxV = (
                        np.sum(NoTxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(NoTxV["new_hpv_screens"] * cost_dict["hpv"])
                        + np.sum(NoTxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(NoTxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(NoTxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(NoTxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_noTxV += cost_noTxV

                    if "intro 2030" in txvx_scen:
                        econ_df_to_use = faster_econ_df
                        txvx_scen_to_use = "Mass TxV, 90/50, age 30"

                    elif "HPV" in txvx_scen:
                        txvx_scen_to_use = "No TxV"
                        screen_scen_label_to_use = txvx_scen

                    else:
                        txvx_scen_to_use = txvx_scen
                        screen_scen_label_to_use = screen_scen_label

                    TxV = sc.dcp(
                        econ_df_to_use[
                            (econ_df_to_use.screen_scen == screen_scen_label_to_use)
                            & (econ_df_to_use.vx_scen == vx_scen_label)
                            & (econ_df_to_use.txvx_scen == txvx_scen_to_use)
                            & (econ_df_to_use.location == location)
                        ][
                            [
                                "dalys",
                                "new_poc_hpv_screens",
                                "new_hpv_screens",
                                "new_vaccinations",
                                "new_tx_vaccinations",
                                "new_thermal_ablations",
                                "new_leeps",
                                "new_cancer_treatments",
                            ]
                        ]
                    )

                    dalys_TxV += TxV["dalys"]
                    cost_TxV = (
                        np.sum(1.8 * TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        # np.sum(TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(TxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(TxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(TxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(TxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_TxV += cost_TxV

                dalys_averted = dalys_noTxV - dalys_TxV
                additional_cost = costs_TxV - costs_noTxV
                cost_daly_averted = additional_cost / dalys_averted
                print(
                    f"{background_scen_label}, {txvx_scen}, averts {int(dalys_averted[0])/1e6} million DALYs at cost/DALY averted: {cost_daly_averted[0]}"
                )

                # print(f"{background_scen_label}, No TxV, {NoTxV}")

                # print(f"{background_scen_label}, {txvx_scen}, {TxV}")
                if it == 0:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color="black",
                        marker=markers[it],
                        linestyle="None",
                        markersize=15,
                        # alpha=0.5,
                        markeredgecolor="black",
                    )

                else:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[1],
                        marker=markers[it],
                        linestyle="None",
                        markersize=15,
                        # alpha=0.5,
                        markeredgecolor="black",
                    )

                handles.append(handle)

    ax.legend(
        [
            handles[0],
            handles[1],
            handles[2],
            handles[3],
            handles[4],
            handles[5],
            handles[6],
        ],
        [
            "Option 3a, 90/50",
            "Option 3a: 70/30",
            "Option 3a: Faster introduction (2030)",
            "Option 3a: Delayed introduction (2038)",
            "Option 3a: No immune memory",
            "Option 3a: 50% cross-protection",
            "Option 2a: POC HPV test (30% LTFU)",
        ],
        title="Product characteristic",
        bbox_to_anchor=(0.9, -0.2),
        ncol=2,
    )
    # ax.set_ylim(top=300)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("Incremental costs/DALY averted,\n$USD 2030-2060")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA_sens_v2{filestem}.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_total_costs(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    filestem=None,
):
    ut.set_font(size=14)
    econdfs = sc.autolist()

    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        econdfs += econdf

    econ_df = pd.concat(econdfs)

    x = np.arange(len(background_scens))  # the label locations
    width = 0.2  # the width of the bars

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]

    txvx_scen_label = {
        "No TxV": "Option 1: Do nothing",
        "Mass TxV, 90/50, age 30": "Option 3a",
        "TnV TxV, 90/50, age 30": "Option 3b",
        "HPV, 35% sc cov, 30% LTFU": "Option 2a",
        "HPV, 70% sc cov, 30% LTFU": "Option 2a",
    }

    colors = sc.gridcolors(10)

    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            if ib == 0 and "HPV" in txvx_scen:
                it -= 1
                pass
            elif "HPV" in txvx_scen and txvx_scen[:-10] != screen_scen_label[:-10]:
                it -= 1
            else:
                total_costs = 0
                if "HPV" in txvx_scen:
                    screen_scen_label_to_use = sc.dcp(txvx_scen)
                    txvx_scen_to_use = "No TxV"
                    it = 3
                else:
                    screen_scen_label_to_use = screen_scen_label
                    txvx_scen_to_use = txvx_scen
                for location in locations:

                    TxV = sc.dcp(
                        econ_df[
                            (econ_df.screen_scen == screen_scen_label_to_use)
                            & (econ_df.vx_scen == vx_scen_label)
                            & (econ_df.txvx_scen == txvx_scen_to_use)
                            & (econ_df.location == location)
                        ][
                            [
                                "dalys",
                                "new_poc_hpv_screens",
                                "new_hpv_screens",
                                "new_vaccinations",
                                "new_tx_vaccinations",
                                "new_thermal_ablations",
                                "new_leeps",
                                "new_cancer_treatments",
                            ]
                        ]
                    )

                    cost_TxV = (
                        np.sum(1.8 * TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(TxV["new_hpv_screens"] * cost_dict["hpv"])
                        + np.sum(TxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(TxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(TxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(TxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    total_costs += cost_TxV
                if ib == 2:

                    ax.bar(
                        xes[it][ib],
                        total_costs,
                        color=colors[it],
                        width=width,
                        edgecolor="black",
                        label=txvx_scen_label[txvx_scen],
                    )
                else:
                    ax.bar(
                        xes[it][ib],
                        total_costs,
                        color=colors[it],
                        width=width,
                        edgecolor="black",
                    )

                ax.text(
                    xes[it][ib],
                    total_costs + 100e6,
                    f"${round(total_costs / 1e9, 1)}",
                    ha="center",
                )

    ax.set_xticks(x + 1.5 * width, background_scens.keys())

    # ax.set_ylim(top=15e9)
    ax.legend(ncol=2)
    ax.set_xlabel("Screen coverage")
    ax.set_ylabel("Total costs (2030-2060)")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/total_cost{filestem}.png"
    fig.savefig(fig_name, dpi=100)
    return


def plot_CEA_sens(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    filestem=None,
):
    ut.set_font(size=14)
    econdfs = sc.autolist()
    fasterecondfs = sc.autolist()

    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        econdfs += econdf

        faster_econdf = sc.loadobj(f"{ut.resfolder}/{location}_feb24_econ.obj")
        fasterecondfs += faster_econdf

    econ_df = pd.concat(econdfs)
    faster_econ_df = pd.concat(fasterecondfs)

    handles = []
    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "D", "h", "X", "o", "d"]
    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            if "HPV" in txvx_scen and txvx_scen[:-10] != screen_scen_label[:-10]:
                pass
            else:
                dalys_noTxV = 0
                dalys_TxV = 0
                costs_noTxV = 0
                costs_TxV = 0
                for location in locations:
                    econ_df_to_use = econ_df
                    NoTxV = sc.dcp(
                        econ_df_to_use[
                            (econ_df_to_use.screen_scen == screen_scen_label)
                            & (econ_df_to_use.vx_scen == vx_scen_label)
                            & (econ_df_to_use.txvx_scen == "No TxV")
                            & (econ_df_to_use.location == location)
                        ][
                            [
                                "dalys",
                                "new_poc_hpv_screens",
                                "new_hpv_screens",
                                "new_vaccinations",
                                "new_tx_vaccinations",
                                "new_thermal_ablations",
                                "new_leeps",
                                "new_cancer_treatments",
                            ]
                        ]
                    )

                    dalys_noTxV += NoTxV["dalys"]

                    cost_noTxV = (
                        np.sum(NoTxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(NoTxV["new_hpv_screens"] * cost_dict["hpv"])
                        + np.sum(NoTxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(NoTxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(NoTxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(NoTxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_noTxV += cost_noTxV

                    if "intro 2030" in txvx_scen:
                        econ_df_to_use = faster_econ_df
                        txvx_scen = "Mass TxV, 90/50, age 30"

                    TxV = sc.dcp(
                        econ_df_to_use[
                            (econ_df_to_use.screen_scen == screen_scen_label)
                            & (econ_df_to_use.vx_scen == vx_scen_label)
                            & (econ_df_to_use.txvx_scen == txvx_scen)
                            & (econ_df_to_use.location == location)
                        ][
                            [
                                "dalys",
                                "new_poc_hpv_screens",
                                "new_hpv_screens",
                                "new_vaccinations",
                                "new_tx_vaccinations",
                                "new_thermal_ablations",
                                "new_leeps",
                                "new_cancer_treatments",
                            ]
                        ]
                    )

                    dalys_TxV += TxV["dalys"]
                    cost_TxV = (
                        np.sum(1.8 * TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        # np.sum(TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(TxV["new_hpv_screens"] * cost_dict["hpv"])
                        + np.sum(TxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(TxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(TxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(TxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_TxV += cost_TxV

                dalys_averted = dalys_noTxV - dalys_TxV
                additional_cost = costs_TxV - costs_noTxV
                cost_daly_averted = additional_cost / dalys_averted
                print(
                    f"{background_scen_label}, {txvx_scen}, averts {int(dalys_averted[0])/1e6} million DALYs at cost/DALY averted: {cost_daly_averted[0]}"
                )

                # print(f"{background_scen_label}, No TxV, {NoTxV}")

                # print(f"{background_scen_label}, {txvx_scen}, {TxV}")
                if it == 0:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color="black",
                        marker=markers[it],
                        linestyle="None",
                        markersize=15,
                        # alpha=0.5,
                        markeredgecolor="black",
                    )

                else:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[1],
                        marker=markers[it],
                        linestyle="None",
                        markersize=15,
                        # alpha=0.5,
                        markeredgecolor="black",
                    )

                handles.append(handle)

    ax.legend(
        [
            handles[0],
            handles[1],
            handles[2],
            handles[3],
            handles[4],
            handles[5],
        ],
        [
            "Baseline",
            "70/30",
            "Faster introduction (2030)",
            "Delayed introduction (2038)",
            "No immune memory",
            "50% cross-protection",
        ],
        title="Product characteristic",
        loc="upper right",
        ncol=2,
    )
    ax.set_ylim([0, 250])
    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("Incremental costs/DALY averted,\n$USD 2030-2060")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA_sens{filestem}.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA_v2(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    filestem=None,
):
    ut.set_font(size=14)
    econdfs = sc.autolist()

    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        econdfs += econdf

    econ_df = pd.concat(econdfs)

    handles = []
    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "D", "^", "x"]
    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            if ib == 0 and "HPV" in txvx_scen:
                pass
            elif "HPV" in txvx_scen and txvx_scen[:-10] != screen_scen_label[:-10]:
                pass
            else:
                dalys_noTxV = 0
                dalys_TxV = 0
                costs_noTxV = 0
                costs_TxV = 0
                for location in locations:
                    econ_df_to_use = econ_df
                    NoTxV = sc.dcp(
                        econ_df_to_use[
                            (econ_df_to_use.screen_scen == screen_scen_label)
                            & (econ_df_to_use.vx_scen == vx_scen_label)
                            & (econ_df_to_use.txvx_scen == "No TxV")
                            & (econ_df_to_use.location == location)
                        ][
                            [
                                "dalys",
                                "new_poc_hpv_screens",
                                "new_hpv_screens",
                                "new_vaccinations",
                                "new_tx_vaccinations",
                                "new_thermal_ablations",
                                "new_leeps",
                                "new_cancer_treatments",
                            ]
                        ]
                    )

                    dalys_noTxV += NoTxV["dalys"]

                    cost_noTxV = (
                        np.sum(NoTxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(NoTxV["new_hpv_screens"] * cost_dict["hpv"])
                        + np.sum(NoTxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(NoTxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(NoTxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(NoTxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_noTxV += cost_noTxV

                    if "HPV" in txvx_scen:
                        screen_scen_label_to_use = sc.dcp(txvx_scen)
                        txvx_scen_to_use = "No TxV"
                        it = 2
                    else:
                        screen_scen_label_to_use = screen_scen_label
                        txvx_scen_to_use = txvx_scen

                    TxV = sc.dcp(
                        econ_df_to_use[
                            (econ_df_to_use.screen_scen == screen_scen_label_to_use)
                            & (econ_df_to_use.vx_scen == vx_scen_label)
                            & (econ_df_to_use.txvx_scen == txvx_scen_to_use)
                            & (econ_df_to_use.location == location)
                        ][
                            [
                                "dalys",
                                "new_poc_hpv_screens",
                                "new_hpv_screens",
                                "new_vaccinations",
                                "new_tx_vaccinations",
                                "new_thermal_ablations",
                                "new_leeps",
                                "new_cancer_treatments",
                            ]
                        ]
                    )

                    dalys_TxV += TxV["dalys"]
                    cost_TxV = (
                        np.sum(1.8 * TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        # np.sum(TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(TxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(TxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(TxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(TxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_TxV += cost_TxV

                dalys_averted = dalys_noTxV - dalys_TxV
                additional_cost = costs_TxV - costs_noTxV
                cost_daly_averted = additional_cost / dalys_averted
                print(
                    f"{background_scen_label}, {txvx_scen}, averts {int(dalys_averted[0])/1e6} million DALYs at cost/DALY averted: {cost_daly_averted[0]}"
                )

                # print(f"{background_scen_label}, No TxV, {NoTxV}")

                # print(f"{background_scen_label}, {txvx_scen}, {TxV}")

                (handle,) = ax.plot(
                    dalys_averted / 1e6,
                    cost_daly_averted,
                    color=colors[ib],
                    marker=markers[it],
                    linestyle="None",
                    markersize=15,
                    # alpha=0.5,
                    markeredgecolor="black",
                )

                handles.append(handle)

    legend1 = ax.legend(
        [handles[0], handles[2], handles[5]],
        background_scens.keys(),
        title="Screen coverage (50% LTFU)",
        bbox_to_anchor=(0.3, -0.2),
        # ncol=3,
    )

    ax.legend(
        [handles[2], handles[3], handles[4]],
        [
            f"Option 3a: Mass TxV",
            f"Option 3b: Test & vaccinate",
            "Option 2a: POC HPV test (30% LTFU)",
        ],
        title="Product characteristic",
        bbox_to_anchor=(1, -0.2),
        # ncol=3,
    )
    ax.axhline(y=0, color="black", linewidth=0.5)
    pl.gca().add_artist(legend1)
    # ax.set_ylim(top=700)
    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("Incremental costs/DALY averted,\n$USD 2030-2060")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA{filestem}.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA_tnv(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    filestem=None,
):

    ut.set_font(size=14)
    econdfs = sc.autolist()
    dfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        econdfs += econdf

        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        dfs += df
    econ_df = pd.concat(econdfs)
    df = pd.concat(dfs)

    handles = []
    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "D", "^", "x"]
    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            if ib == 0 and "HPV" in txvx_scen:
                pass
            elif "HPV" in txvx_scen and txvx_scen[:-10] != screen_scen_label[:-10]:
                pass
            else:
                dalys_noTxV = 0
                dalys_TxV = 0
                costs_noTxV = 0
                costs_TxV = 0
                for location in locations:
                    econ_df_to_use = econ_df
                    df_to_use = df
                    NoTxV_econdf_counts = econ_df_to_use[
                        (econ_df_to_use.screen_scen == screen_scen_label)
                        & (econ_df_to_use.vx_scen == vx_scen_label)
                        & (econ_df_to_use.txvx_scen == "No TxV")
                        & (econ_df_to_use.location == location)
                    ][["dalys"]].sum()

                    NoTxV = df_to_use[
                        (df_to_use.screen_scen == screen_scen_label)
                        & (df_to_use.vx_scen == vx_scen_label)
                        & (df_to_use.txvx_scen == "No TxV")
                        & (df_to_use.location == location)
                    ][
                        [
                            "new_txvx_doses",
                            "new_screens",
                            "new_cin_treatments",
                            "new_cancer_treatments",
                        ]
                    ].sum()

                    dalys_noTxV += NoTxV_econdf_counts["dalys"]

                    cost_noTxV = (
                        np.sum(NoTxV["new_txvx_doses"] * cost_dict["txv"])
                        + np.sum(NoTxV["new_screens"] * cost_dict["hpv"])
                        + np.sum(0.1 * NoTxV["new_cin_treatments"] * cost_dict["leep"])
                        + np.sum(
                            0.9 * NoTxV["new_cin_treatments"] * cost_dict["ablation"]
                        )
                        + np.sum(NoTxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_noTxV += cost_noTxV

                    if "HPV" in txvx_scen:
                        screen_scen_label_to_use = sc.dcp(txvx_scen)
                        txvx_scen_to_use = "No TxV"
                        it = 4
                    else:
                        screen_scen_label_to_use = screen_scen_label
                        txvx_scen_to_use = txvx_scen

                    TxV_econdf_counts = econ_df_to_use[
                        (econ_df_to_use.screen_scen == screen_scen_label_to_use)
                        & (econ_df_to_use.vx_scen == vx_scen_label)
                        & (econ_df_to_use.txvx_scen == txvx_scen_to_use)
                        & (econ_df_to_use.location == location)
                    ][["dalys"]].sum()

                    TxV = df_to_use[
                        (df_to_use.screen_scen == screen_scen_label_to_use)
                        & (df_to_use.vx_scen == vx_scen_label)
                        & (df_to_use.txvx_scen == txvx_scen_to_use)
                        & (df_to_use.location == location)
                    ][
                        [
                            "new_txvx_doses",
                            "new_screens",
                            "new_cin_treatments",
                            "new_cancer_treatments",
                        ]
                    ].sum()

                    dalys_TxV += TxV_econdf_counts["dalys"]
                    if "TnV" in txvx_scen_to_use or "HPV" in txvx_scen_to_use:
                        hpv_cost = cost_dict["poc_hpv"]
                    else:
                        hpv_cost = cost_dict["hpv"]
                    cost_TxV = (
                        np.sum(TxV["new_txvx_doses"] * cost_dict["txv"])
                        + np.sum(TxV["new_screens"] * hpv_cost)
                        + np.sum(0.1 * TxV["new_cin_treatments"] * cost_dict["leep"])
                        + np.sum(
                            0.9 * TxV["new_cin_treatments"] * cost_dict["ablation"]
                        )
                        + np.sum(TxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    costs_TxV += cost_TxV

                dalys_averted = dalys_noTxV - dalys_TxV
                additional_cost = costs_TxV - costs_noTxV
                cost_daly_averted = additional_cost / dalys_averted
                print(
                    f"{background_scen_label}, {txvx_scen}, averts {dalys_averted} DALYs at cost/DALY averted: {cost_daly_averted}"
                )

                print(f"{background_scen_label}, No TxV, {NoTxV}")

                print(f"{background_scen_label}, {txvx_scen}, {TxV}")

                (handle,) = ax.plot(
                    dalys_averted / 1e6,
                    cost_daly_averted,
                    color=colors[ib],
                    marker=markers[it],
                    linestyle="None",
                    markersize=15,
                    # alpha=0.5,
                    markeredgecolor="black",
                )

                handles.append(handle)

    legend1 = ax.legend(
        [handles[0], handles[4], handles[9]],
        background_scens.keys(),
        title="Screen coverage (50% LTFU)",
        bbox_to_anchor=(1, 0.8),
        ncol=3,
    )

    ax.legend(
        [handles[4], handles[5], handles[6], handles[7], handles[8]],
        [
            "Mass TxV, 90/50",
            "Mass TxV, 70/30",
            "Test & vaccinate, 90/50",
            "Test & vaccinate, 70/30",
            "POC HPV test (30% LTFU)",
        ],
        title="Product characteristic",
        loc="upper right",
        ncol=3,
    )
    ax.axhline(y=0, color="black", linewidth=0.5)
    pl.gca().add_artist(legend1)
    ax.set_ylim(top=500)
    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("Incremental costs/DALY averted,\n$USD 2030-2060")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA_hpv_sens{filestem}.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA_PDVAC(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    hpv_test_cost_range=None,
    filestem=None,
):

    ut.set_font(size=14)
    econdfs = sc.autolist()
    delayed_intro_econ_dfs = sc.autolist()

    dfs = sc.autolist()
    delayed_intro_dfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        delayed_econ_introdf = sc.loadobj(
            f"{ut.resfolder}/{location}_delayed_intro_econ.obj"
        )
        delayed_intro_econ_dfs += delayed_econ_introdf
        econdfs += econdf

        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        delayed_introdf = sc.loadobj(f"{ut.resfolder}/{location}_delayed_intro.obj")
        delayed_intro_dfs += delayed_introdf
        dfs += df
    econ_df = pd.concat(econdfs)
    delayed_intro_econ_df = pd.concat(delayed_intro_econ_dfs)
    df = pd.concat(dfs)
    delayed_intro_df = pd.concat(delayed_intro_dfs)

    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "D", "^", "x"]
    marker_sizes = [10, 15, 20, 25]
    handles = []

    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            for i_cost, hpv_test_cost in enumerate(hpv_test_cost_range):
                dalys_noTxV = 0
                dalys_TxV = 0
                costs_noTxV = 0
                costs_TxV = 0
                for location in locations:
                    if "intro 2035" in txvx_scen:
                        econ_df_to_use = delayed_intro_econ_df
                        df_to_use = delayed_intro_df
                    else:
                        econ_df_to_use = econ_df
                        df_to_use = df
                    NoTxV_econdf_counts = econ_df_to_use[
                        (econ_df_to_use.screen_scen == screen_scen_label)
                        & (econ_df_to_use.vx_scen == vx_scen_label)
                        & (econ_df_to_use.txvx_scen == "No TxV")
                        & (econ_df_to_use.location == location)
                    ][["dalys"]].sum()

                    NoTxV = (
                        df_to_use[
                            (df_to_use.screen_scen == screen_scen_label)
                            & (df_to_use.vx_scen == vx_scen_label)
                            & (df_to_use.txvx_scen == "No TxV")
                            & (df_to_use.location == location)
                        ]
                        .groupby("year")[
                            [
                                "new_txvx_doses",
                                "new_screens",
                                "new_cin_treatments",
                                "new_cancer_treatments",
                            ]
                        ]
                        .sum()
                    )

                    dalys_noTxV += NoTxV_econdf_counts["dalys"]

                    cost_noTxV = (
                        np.sum(NoTxV["new_txvx_doses"].values * cost_dict["txv"])
                        + np.sum(NoTxV["new_screens"].values * hpv_test_cost)
                        + np.sum(
                            0.1 * NoTxV["new_cin_treatments"].values * cost_dict["leep"]
                        )
                        + np.sum(
                            0.9
                            * NoTxV["new_cin_treatments"].values
                            * cost_dict["ablation"]
                        )
                        + np.sum(
                            NoTxV["new_cancer_treatments"].values * cost_dict["cancer"]
                        )
                    )
                    costs_noTxV += cost_noTxV

                    TxV_econdf_counts = econ_df_to_use[
                        (econ_df_to_use.screen_scen == screen_scen_label)
                        & (econ_df_to_use.vx_scen == vx_scen_label)
                        & (econ_df_to_use.txvx_scen == txvx_scen)
                        & (econ_df_to_use.location == location)
                    ][["dalys"]].sum()

                    TxV = (
                        df_to_use[
                            (df_to_use.screen_scen == screen_scen_label)
                            & (df_to_use.vx_scen == vx_scen_label)
                            & (df_to_use.txvx_scen == txvx_scen)
                            & (df_to_use.location == location)
                        ]
                        .groupby("year")[
                            [
                                "new_txvx_doses",
                                "new_screens",
                                "new_cin_treatments",
                                "new_cancer_treatments",
                            ]
                        ]
                        .sum()
                    )

                    dalys_TxV += TxV_econdf_counts["dalys"]

                    cost_TxV = (
                        np.sum(TxV["new_txvx_doses"].values * cost_dict["txv"])
                        + np.sum(TxV["new_screens"].values * hpv_test_cost)
                        + np.sum(
                            0.1 * TxV["new_cin_treatments"].values * cost_dict["leep"]
                        )
                        + np.sum(
                            0.9
                            * TxV["new_cin_treatments"].values
                            * cost_dict["ablation"]
                        )
                        + np.sum(
                            TxV["new_cancer_treatments"].values * cost_dict["cancer"]
                        )
                    )
                    costs_TxV += cost_TxV

                dalys_averted = dalys_noTxV - dalys_TxV
                additional_cost = costs_TxV - costs_noTxV
                cost_daly_averted = additional_cost / dalys_averted

                if "TnV TxV" in txvx_scen:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[ib],
                        marker=markers[it],
                        linestyle="None",
                        markersize=marker_sizes[i_cost],
                        markeredgecolor="black",
                    )
                elif i_cost == 0:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[ib],
                        marker=markers[it],
                        linestyle="None",
                        markersize=marker_sizes[i_cost],
                        markeredgecolor="black",
                    )
                else:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[ib],
                        marker=markers[it],
                        linestyle="None",
                        markersize=marker_sizes[i_cost],
                        alpha=0.2,
                        markeredgecolor="black",
                    )

                handles.append(handle)

    legend1 = ax.legend(
        handles[4:8],
        hpv_test_cost_range,
        title="HPV test cost ($USD)",
        bbox_to_anchor=(0.5, 1),
        ncol=2,
    )
    legend2 = ax.legend(
        [handles[0], handles[4]],
        ["Mass TxV (90/50)", "Test & vaccinate (90/50)"],
        title="TxV scenario",
        loc="upper right",
        ncol=3,
    )
    ax.legend(
        [handles[0], handles[8]],
        background_scens.keys(),
        title="Background intervention scale-up (Pxv-Sc-Tx)",
        bbox_to_anchor=(1.0, 0.8),
        ncol=3,
    )
    pl.gca().add_artist(legend1)
    pl.gca().add_artist(legend2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("Incremental costs/DALY averted,\n$USD 2030-2060")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA_PDVAC.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA_DEG(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    hpv_test_cost_range=None,
    filestem=None,
):

    ut.set_font(size=14)
    econdfs = sc.autolist()
    delayed_intro_econ_dfs = sc.autolist()

    dfs = sc.autolist()
    delayed_intro_dfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        delayed_econ_introdf = sc.loadobj(
            f"{ut.resfolder}/{location}_delayed_intro_econ.obj"
        )
        delayed_intro_econ_dfs += delayed_econ_introdf
        econdfs += econdf

        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        delayed_introdf = sc.loadobj(f"{ut.resfolder}/{location}_delayed_intro.obj")
        delayed_intro_dfs += delayed_introdf
        dfs += df
    econ_df = pd.concat(econdfs)
    delayed_intro_econ_df = pd.concat(delayed_intro_econ_dfs)
    df = pd.concat(dfs)
    delayed_intro_df = pd.concat(delayed_intro_dfs)

    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "D", "^", "x"]
    marker_sizes = [10, 15, 20, 25]
    handles = []

    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            for i_cost, hpv_test_cost in enumerate(hpv_test_cost_range):
                dalys_noTxV = 0
                dalys_TxV = 0
                costs_noTxV = 0
                costs_TxV = 0
                for location in locations:
                    if "intro 2035" in txvx_scen:
                        econ_df_to_use = delayed_intro_econ_df
                        df_to_use = delayed_intro_df
                    else:
                        econ_df_to_use = econ_df
                        df_to_use = df
                    NoTxV_econdf_counts = econ_df_to_use[
                        (econ_df_to_use.screen_scen == screen_scen_label)
                        & (econ_df_to_use.vx_scen == vx_scen_label)
                        & (econ_df_to_use.txvx_scen == "No TxV")
                        & (econ_df_to_use.location == location)
                    ][["dalys"]].sum()

                    NoTxV = (
                        df_to_use[
                            (df_to_use.screen_scen == screen_scen_label)
                            & (df_to_use.vx_scen == vx_scen_label)
                            & (df_to_use.txvx_scen == "No TxV")
                            & (df_to_use.location == location)
                        ]
                        .groupby("year")[
                            [
                                "new_txvx_doses",
                                "new_screens",
                                "new_cin_treatments",
                                "new_cancer_treatments",
                            ]
                        ]
                        .sum()
                    )

                    dalys_noTxV += NoTxV_econdf_counts["dalys"]

                    cost_noTxV = (
                        np.sum(NoTxV["new_txvx_doses"].values * cost_dict["txv"])
                        + np.sum(NoTxV["new_screens"].values * hpv_test_cost)
                        + np.sum(
                            0.1 * NoTxV["new_cin_treatments"].values * cost_dict["leep"]
                        )
                        + np.sum(
                            0.9
                            * NoTxV["new_cin_treatments"].values
                            * cost_dict["ablation"]
                        )
                        + np.sum(
                            NoTxV["new_cancer_treatments"].values * cost_dict["cancer"]
                        )
                    )
                    costs_noTxV += cost_noTxV

                    TxV_econdf_counts = econ_df_to_use[
                        (econ_df_to_use.screen_scen == screen_scen_label)
                        & (econ_df_to_use.vx_scen == vx_scen_label)
                        & (econ_df_to_use.txvx_scen == txvx_scen)
                        & (econ_df_to_use.location == location)
                    ][["dalys"]].sum()

                    TxV = (
                        df_to_use[
                            (df_to_use.screen_scen == screen_scen_label)
                            & (df_to_use.vx_scen == vx_scen_label)
                            & (df_to_use.txvx_scen == txvx_scen)
                            & (df_to_use.location == location)
                        ]
                        .groupby("year")[
                            [
                                "new_txvx_doses",
                                "new_screens",
                                "new_cin_treatments",
                                "new_cancer_treatments",
                            ]
                        ]
                        .sum()
                    )

                    dalys_TxV += TxV_econdf_counts["dalys"]

                    cost_TxV = (
                        np.sum(TxV["new_txvx_doses"].values * cost_dict["txv"])
                        + np.sum(TxV["new_screens"].values * hpv_test_cost)
                        + np.sum(
                            0.1 * TxV["new_cin_treatments"].values * cost_dict["leep"]
                        )
                        + np.sum(
                            0.9
                            * TxV["new_cin_treatments"].values
                            * cost_dict["ablation"]
                        )
                        + np.sum(
                            TxV["new_cancer_treatments"].values * cost_dict["cancer"]
                        )
                    )
                    costs_TxV += cost_TxV

                dalys_averted = dalys_noTxV - dalys_TxV
                additional_cost = costs_TxV - costs_noTxV
                cost_daly_averted = additional_cost / dalys_averted

                if "TnV TxV" in txvx_scen:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[ib],
                        marker=markers[it],
                        linestyle="None",
                        markersize=marker_sizes[i_cost],
                        markeredgecolor="black",
                    )
                elif i_cost == 0:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[ib],
                        marker=markers[it],
                        linestyle="None",
                        markersize=marker_sizes[i_cost],
                        markeredgecolor="black",
                    )
                else:
                    (handle,) = ax.plot(
                        dalys_averted / 1e6,
                        cost_daly_averted,
                        color=colors[ib],
                        marker=markers[it],
                        linestyle="None",
                        markersize=marker_sizes[i_cost],
                        alpha=0.2,
                        markeredgecolor="black",
                    )

                handles.append(handle)

    legend1 = ax.legend(
        handles[4:8],
        hpv_test_cost_range,
        title="HPV test cost ($USD)",
        bbox_to_anchor=(0.67, 1),
        ncol=2,
    )
    legend2 = ax.legend(
        [handles[0], handles[4]],
        ["Mass TxV (90/50)", "Test & vaccinate (90/50)"],
        title="TxV scenario",
        loc="upper right",
        # ncol=3,
    )
    ax.legend(
        [handles[0], handles[8]],
        background_scens.keys(),
        title="Background intervention scale-up (Pxv-Sc-Tx)",
        bbox_to_anchor=(1.0, 0.8),
        ncol=3,
    )
    pl.gca().add_artist(legend1)
    pl.gca().add_artist(legend2)

    ax.set_ylim(bottom=0)
    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("Incremental costs/DALY averted,\n$USD 2030-2060")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA_DEG.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA_SnT(
    locations=None,
    background_scens=None,
    txvx_scens=None,
    filestem=None,
):

    ut.set_font(size=14)
    econdfs = sc.autolist()
    delayed_intro_econ_dfs = sc.autolist()

    dfs = sc.autolist()
    delayed_intro_dfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        delayed_econ_introdf = sc.loadobj(
            f"{ut.resfolder}/{location}_delayed_intro_econ.obj"
        )
        delayed_intro_econ_dfs += delayed_econ_introdf
        econdfs += econdf

        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        delayed_introdf = sc.loadobj(f"{ut.resfolder}/{location}_delayed_intro.obj")
        delayed_intro_dfs += delayed_introdf
        dfs += df
    econ_df = pd.concat(econdfs)
    delayed_intro_econ_df = pd.concat(delayed_intro_econ_dfs)
    df = pd.concat(dfs)
    delayed_intro_df = pd.concat(delayed_intro_dfs)

    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "D", "^", "x"]
    marker_sizes = [10, 15, 20, 25]
    handles = []

    baseline_dalys = 0
    for location in locations:
        baseline_dalys += econ_df[
            (econ_df.screen_scen == "No screening")
            & (econ_df.vx_scen == "Vx, 70% cov, 9-14")
            & (econ_df.txvx_scen == "No TxV")
            & (econ_df.location == location)
        ][["dalys"]].sum()

    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        if ib > 0:
            vx_scen_label = background_scen["vx_scen"]
            screen_scen_label = background_scen["screen_scen"]
            for it, txvx_scen in enumerate(txvx_scens):
                dalys = 0
                costs = 0

                for location in locations:
                    econdf_counts = econ_df[
                        (econ_df.screen_scen == screen_scen_label)
                        & (econ_df.vx_scen == vx_scen_label)
                        & (econ_df.txvx_scen == txvx_scen)
                        & (econ_df.location == location)
                    ][["dalys"]].sum()

                    SnT = (
                        df[
                            (df.screen_scen == screen_scen_label)
                            & (df.vx_scen == vx_scen_label)
                            & (df.txvx_scen == txvx_scen)
                            & (df.location == location)
                        ]
                        .groupby("year")[
                            [
                                "new_txvx_doses",
                                "new_screens",
                                "new_cin_treatments",
                                "new_cancer_treatments",
                            ]
                        ]
                        .sum()
                    )

                    dalys += econdf_counts["dalys"]

                    cost = (
                        np.sum(SnT["new_txvx_doses"].values * cost_dict["txv"])
                        + np.sum(SnT["new_screens"].values * cost_dict["hpv"])
                        + np.sum(
                            0.1 * SnT["new_cin_treatments"].values * cost_dict["leep"]
                        )
                        + np.sum(
                            0.9
                            * SnT["new_cin_treatments"].values
                            * cost_dict["ablation"]
                        )
                        + np.sum(
                            SnT["new_cancer_treatments"].values * cost_dict["cancer"]
                        )
                    )
                    costs += cost

                (handle,) = ax.plot(
                    (baseline_dalys - dalys) / 1e6,
                    costs / (baseline_dalys - dalys),
                    color=colors[ib],
                    linestyle="None",
                    marker=markers[it],
                    markersize=15,
                    markeredgecolor="black",
                    label=background_scen_label,
                )

                handles.append(handle)

    # legend1 = ax.legend(
    #     handles[0:3],
    #     txvx_scens,
    #     title="TxV scenario",
    #     bbox_to_anchor=(0.35, 0.98),
    # )

    ax.legend(
        title="S&T scale-up (Pxv-Sc-Tx)",
        # bbox_to_anchor=(0.3, 0.7),
    )
    # pl.gca().add_artist(legend1)

    ax.set_ylim([0, 150])
    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("ICER")
    # sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA_SnT.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA(locations=None, background_scens=None, txvx_scens=None, filestem=None):

    ut.set_font(size=14)
    econdfs = sc.autolist()
    delayed_intro_econ_dfs = sc.autolist()

    dfs = sc.autolist()
    delayed_intro_dfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}_econ.obj")
        delayed_econ_introdf = sc.loadobj(
            f"{ut.resfolder}/{location}_delayed_intro_econ.obj"
        )
        delayed_intro_econ_dfs += delayed_econ_introdf
        econdfs += econdf

        df = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        delayed_introdf = sc.loadobj(f"{ut.resfolder}/{location}_delayed_intro.obj")
        delayed_intro_dfs += delayed_introdf
        dfs += df
    econ_df = pd.concat(econdfs)
    delayed_intro_econ_df = pd.concat(delayed_intro_econ_dfs)
    df = pd.concat(dfs)
    delayed_intro_df = pd.concat(delayed_intro_dfs)

    colors = ["orange", "red", "darkred"]
    markers = ["s", "v", "P", "*", "+", "D", "^", "x"]
    handles = []

    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(
        background_scens.items()
    ):
        vx_scen_label = background_scen["vx_scen"]
        screen_scen_label = background_scen["screen_scen"]
        for it, txvx_scen in enumerate(txvx_scens):
            dalys_noTxV = 0
            dalys_TxV = 0
            costs_noTxV = 0
            costs_TxV = 0
            for location in locations:
                if "intro 2035" in txvx_scen:
                    econ_df_to_use = delayed_intro_econ_df
                    df_to_use = delayed_intro_df
                else:
                    econ_df_to_use = econ_df
                    df_to_use = df
                NoTxV_econdf_counts = econ_df_to_use[
                    (econ_df_to_use.screen_scen == screen_scen_label)
                    & (econ_df_to_use.vx_scen == vx_scen_label)
                    & (econ_df_to_use.txvx_scen == "No TxV")
                    & (econ_df_to_use.location == location)
                ][["dalys"]].sum()

                NoTxV = (
                    df_to_use[
                        (df_to_use.screen_scen == screen_scen_label)
                        & (df_to_use.vx_scen == vx_scen_label)
                        & (df_to_use.txvx_scen == "No TxV")
                        & (df_to_use.location == location)
                    ]
                    .groupby("year")[
                        [
                            "new_txvx_doses",
                            "new_screens",
                            "new_cin_treatments",
                            "new_cancer_treatments",
                        ]
                    ]
                    .sum()
                )

                dalys_noTxV += NoTxV_econdf_counts["dalys"]

                cost_noTxV = (
                    np.sum(NoTxV["new_txvx_doses"].values * cost_dict["txv"])
                    + np.sum(NoTxV["new_screens"].values * cost_dict["hpv"])
                    + np.sum(
                        0.1 * NoTxV["new_cin_treatments"].values * cost_dict["leep"]
                    )
                    + np.sum(
                        0.9 * NoTxV["new_cin_treatments"].values * cost_dict["ablation"]
                    )
                    + np.sum(
                        NoTxV["new_cancer_treatments"].values * cost_dict["cancer"]
                    )
                )
                costs_noTxV += cost_noTxV

                TxV_econdf_counts = econ_df_to_use[
                    (econ_df_to_use.screen_scen == screen_scen_label)
                    & (econ_df_to_use.vx_scen == vx_scen_label)
                    & (econ_df_to_use.txvx_scen == txvx_scen)
                    & (econ_df_to_use.location == location)
                ][["dalys"]].sum()

                TxV = (
                    df_to_use[
                        (df_to_use.screen_scen == screen_scen_label)
                        & (df_to_use.vx_scen == vx_scen_label)
                        & (df_to_use.txvx_scen == txvx_scen)
                        & (df_to_use.location == location)
                    ]
                    .groupby("year")[
                        [
                            "new_txvx_doses",
                            "new_screens",
                            "new_cin_treatments",
                            "new_cancer_treatments",
                        ]
                    ]
                    .sum()
                )

                dalys_TxV += TxV_econdf_counts["dalys"]

                cost_TxV = (
                    np.sum(TxV["new_txvx_doses"].values * cost_dict["txv"])
                    + np.sum(TxV["new_screens"].values * cost_dict["hpv"])
                    + np.sum(0.1 * TxV["new_cin_treatments"].values * cost_dict["leep"])
                    + np.sum(
                        0.9 * TxV["new_cin_treatments"].values * cost_dict["ablation"]
                    )
                    + np.sum(TxV["new_cancer_treatments"].values * cost_dict["cancer"])
                )
                costs_TxV += cost_TxV

            dalys_averted = dalys_noTxV - dalys_TxV
            additional_cost = costs_TxV - costs_noTxV
            cost_daly_averted = additional_cost / dalys_averted

            (handle,) = ax.plot(
                dalys_averted / 1e6,
                cost_daly_averted,
                color=colors[ib],
                marker=markers[it],
                linestyle="None",
                markersize=15,
            )
            handles.append(handle)

    ax.set_ylim(bottom=0)
    legend1 = ax.legend(
        handles[0:7],
        [
            "90/50",
            "70/30",
            "Delayed intro",
            "50% cross-protection",
            "No immune memory",
            "Test & vaccinate (90/50)",
            "Test & vaccinate (70/30)",
        ],
        title="TxV scenario",
        loc="upper right",
        ncol=3,
    )
    ax.legend(
        [handles[0], handles[7], handles[14]],
        background_scens.keys(),
        title="Background intervention scale-up",
        bbox_to_anchor=(1.0, 0.75),
    )
    pl.gca().add_artist(legend1)

    ax.set_xlabel("DALYs averted (millions), 2030-2060")
    ax.set_ylabel("Incremental costs/DALY averted,\n$USD 2030-2060")
    sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CEA{filestem}.png"
    fig.savefig(fig_name, dpi=100)

    return


def plot_age_causal(locations, scens, txv_scens, filestem):

    ut.set_font(size=24)
    colors = sc.gridcolors(10)
    x = np.arange(len(scens))  # the label locations
    width = 0.2  # the width of the bars

    age_causal_df = sc.loadobj(f"results/natural_history.obj")

    r1 = np.arange(len(scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]

    xes = [r1, r2, r3, r4]
    fig, axes = pl.subplots(
        nrows=2, ncols=len(locations), figsize=(16, 12), sharex=False, sharey=False
    )
    for il, location in enumerate(locations):
        ax = axes[0, il]
        ax.set_title(location.capitalize())
        age_causal_combined = pd.concat(age_causal_df)
        age_causal = age_causal_combined[age_causal_combined["location"] == location]
        sns.violinplot(x="Health event", y="Age", data=age_causal, ax=ax)
        ax = axes[1, il]

        bigdf = sc.loadobj(f"{ut.resfolder}/{location}{filestem}.obj")
        for ib, (ib_label, ib_scens) in enumerate(scens.items()):
            for it, (txvx_scen, txvx_scen_label) in enumerate(txv_scens.items()):
                vx_scen_label = ib_scens[0]
                screen_scen_label = ib_scens[1]
                df = (
                    bigdf[
                        (bigdf.screen_scen == screen_scen_label)
                        & (bigdf.vx_scen == vx_scen_label)
                        & (bigdf.txvx_scen == txvx_scen_label)
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
                    .sum()[2030:]
                )

                years = np.array(df.index)
                cum_cases = np.sum(df["cancers"])

                if txvx_scen_label == "No TxV":
                    color = colors[0]
                    txvx_scen_label_to_plot = txvx_scen_label
                else:
                    color = colors[it + 2]
                    txvx_scen_label_to_plot = txvx_scen
                if ib == 0:
                    ax.bar(
                        xes[it][ib],
                        cum_cases,
                        width,
                        color=color,
                        edgecolor="black",
                        label=txvx_scen_label_to_plot,
                    )
                else:
                    ax.bar(
                        xes[it][ib],
                        cum_cases,
                        width,
                        color=color,
                        edgecolor="black",
                    )

                # if location == "india":
                #     ax.text(
                #         xes[it][ib],
                #         cum_cases + 10e4,
                #         round(cum_cases / 1e6, 1),
                #         ha="center",
                #     )
                # else:
                #     ax.text(
                #         xes[it][ib],
                #         cum_cases + 10e3,
                #         round(cum_cases / 1e3, 1),
                #         ha="center",
                #     )

    axes[1, 0].set_ylabel("Cervical cancer cases (2030-2060)")
    axes[1, 0].set_xticks(x + width, scens.keys())
    axes[1, 1].set_xticks(x + width, scens.keys())

    axes[1, 0].set_xlabel("Background intervention scenario (PxV-Sc-Tx)")
    axes[1, 1].set_xlabel("Background intervention scenario (PxV-Sc-Tx)")
    sc.SIticks(axes[1, 0])
    sc.SIticks(axes[1, 1])
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CC_burden_country_comparison.png"
    sc.savefig(fig_name, dpi=100)

    return


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
    ]
    
    plot_fig1_v2(
        locations=locations,
        scens=['Status quo', 'Increase screening', 'HPV FASTER'],
        filestem="_june6",
    )

    plot_total_costs(
        locations=locations,
        background_scens={
            "None": {"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
            "35%": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 35% sc cov, 50% LTFU",
            },
            "70%": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 70% sc cov, 50% LTFU",
            },
        },
        txvx_scens=[
            "No TxV",
            "Mass TxV, 90/50, age 30",
            # "Mass TxV, 70/30, age 30",
            "TnV TxV, 90/50, age 30",
            # "TnV TxV, 70/30, age 30",
            "HPV, 35% sc cov, 30% LTFU",
            "HPV, 70% sc cov, 30% LTFU",
        ],
        filestem="_feb25",
    )

    plot_burden_redux(
        locations=locations,
        background_scen=[
            "Vx, 70% cov, 9-14",
            "HPV, 35% sc cov, 50% LTFU",
        ],
    )

    plot_CEA_v2(
        locations=locations,
        background_scens={
            "None": {"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
            "35%": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 35% sc cov, 50% LTFU",
            },
            "70%": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 70% sc cov, 50% LTFU",
            },
        },
        txvx_scens=[
            "Mass TxV, 90/50, age 30",
            # "Mass TxV, 70/30, age 30",
            "TnV TxV, 90/50, age 30",
            # "TnV TxV, 70/30, age 30",
            "HPV, 35% sc cov, 30% LTFU",
            "HPV, 70% sc cov, 30% LTFU",
        ],
        filestem="_feb25",
    )

    plot_CEA_sens(
        locations=locations,
        background_scens={
            # "None": {"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
            "35%": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 35% sc cov, 50% LTFU",
            },
            # "70%": {
            #     "vx_scen": "Vx, 70% cov, 9-14",
            #     "screen_scen": "HPV, 70% sc cov, 50% LTFU",
            # },
        },
        txvx_scens=[
            "Mass TxV, 90/50, age 30",
            "Mass TxV, 70/30, age 30",
            "Mass TxV, 90/50, age 30, intro 2030",
            "Mass TxV, 90/50, age 30, intro 2038",
            "Mass TxV, 90/50, age 30, no durable immunity",
            "Mass TxV, 90/50, age 30, cross-protection",
        ],
        filestem="_feb25",
    )

    plot_CEA_sensv2(
        locations=locations,
        background_scens={
            # "None": {"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
            "35%": {
                "vx_scen": "Vx, 70% cov, 9-14",
                "screen_scen": "HPV, 35% sc cov, 50% LTFU",
            },
            # "70%": {
            #     "vx_scen": "Vx, 70% cov, 9-14",
            #     "screen_scen": "HPV, 70% sc cov, 50% LTFU",
            # },
        },
        txvx_scens=[
            "Mass TxV, 90/50, age 30",
            "Mass TxV, 70/30, age 30",
            "Mass TxV, 90/50, age 30, intro 2030",
            "Mass TxV, 90/50, age 30, intro 2038",
            "Mass TxV, 90/50, age 30, no durable immunity",
            "Mass TxV, 90/50, age 30, cross-protection",
            "HPV, 35% sc cov, 30% LTFU",
        ],
        filestem="_feb25",
    )

    # plot_CEA_tnv(
    #     locations=locations,
    #     background_scens={
    #         "None": {"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
    #         "35%": {
    #             "vx_scen": "Vx, 70% cov, 9-14",
    #             "screen_scen": "HPV, 35% sc cov, 50% LTFU",
    #         },
    #         "70%": {
    #             "vx_scen": "Vx, 70% cov, 9-14",
    #             "screen_scen": "HPV, 70% sc cov, 50% LTFU",
    #         },
    #     },
    #     txvx_scens=[
    #         "Mass TxV, 90/50, age 30",
    #         "Mass TxV, 70/30, age 30",
    #         # "Mass TxV, 90/50, age 30, intro 2035",
    #         "TnV TxV, 90/50, age 30",
    #         "TnV TxV, 70/30, age 30",
    #         "HPV, 35% sc cov, 30% LTFU",
    #         "HPV, 70% sc cov, 30% LTFU",
    #     ],
    #     filestem="_feb23",
    # )



    # plot_fig1(
    #     locations=locations,
    #     scens={
    #         "70-0-0": ["Vx, 70% cov, 9-14", "No screening"],
    #         "70-35-50": ["Vx, 70% cov, 9-14", "HPV, 35% sc cov, 50% LTFU"],
    #         "70-35-70": ["Vx, 70% cov, 9-14", "HPV, 35% sc cov, 30% LTFU"],
    #         "70-70-50": ["Vx, 70% cov, 9-14", "HPV, 70% sc cov, 50% LTFU"],
    #         "70-70-70": ["Vx, 70% cov, 9-14", "HPV, 70% sc cov, 30% LTFU"],
    #     },
    #     txv_scens={
    #         "No TxV": "No TxV",
    #         "90/50 TxV, 20% cov": "Mass TxV, 90/50, age 30, 20 cov",
    #         "90/50 TxV, 50% cov": "Mass TxV, 90/50, age 30, 50 cov",
    #         "90/50 TxV, 70% cov": "Mass TxV, 90/50, age 30",
    #     },
    #     filestem="_feb23",
    # )

    plot_age_causal(
        locations=["nigeria", "india"],
        scens={
            "70-0-0": ["Vx, 70% cov, 9-14", "No screening"],
            "70-35-50": ["Vx, 70% cov, 9-14", "HPV, 35% sc cov, 50% LTFU"],
            "70-70-50": ["Vx, 70% cov, 9-14", "HPV, 70% sc cov, 50% LTFU"],
            # "70-35-70": ["Vx, 70% cov, 9-14", "HPV, 35% sc cov, 30% LTFU"],
            # "70-70-70": ["Vx, 70% cov, 9-14", "HPV, 70% sc cov, 30% LTFU"],
        },
        txv_scens={
            "No TxV": "No TxV",
            "90/30 TxV, age 30": "Mass TxV, 90/50, age 30",
            "90/50 TxV, age 35": "Mass TxV, 90/50, age 35",
            "90/50 TxV, age 40": "Mass TxV, 90/50, age 40",
        },
        filestem="_feb25",
    )

    plot_fig2(
        locations=locations,
        background_scen={"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
        txvx_efficacy="90/50",
        txvx_ages=["30", "35", "40"],
        sensitivities=[
            ", no durable immunity",
            ", intro 2038",
            "70/30",
            ", cross-protection",
        ],
        filestem="_feb25",
    )

    # plot_fig2_PDVAC(
    #     locations=locations,
    #     background_scen={'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
    #     txvx_efficacy='90/50',
    #     txvx_age='30',
    #     sensitivities=[', cross-protection',
    #                    ', intro 2035',
    #                    ', no durable immunity',
    #                    '70/30',
    #                    'TnV TxV, 90/50'
    #                    ],
    #     filestem='_nov27'
    # )

    # plot_tnv_sens(
    #     locations=locations,
    #     background_scen={"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
    #     txvx_age="30",
    #     txvx_efficacies=[
    #         "90/50",
    #         #    '50/50',
    #         "70/30",
    #         #    '90/0',
    #     ],
    #     filestem="_feb22",
    # )

    # plot_CEA(
    #     locations=locations,
    #     background_scens={

    #             '90-0-0': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
    #             '90-35-70': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 35% sc cov'},
    #             '90-70-90': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 70% sc cov, 90% tx cov'},
    #         },

    #     txvx_scens=[
    #             'Mass TxV, 90/50, age 30',
    #             'Mass TxV, 70/30, age 30',
    #             'Mass TxV, 90/50, age 30, intro 2035',
    #             'Mass TxV, 90/50, age 30, cross-protection',
    #             'Mass TxV, 90/50, age 30, no durable immunity',
    #             'TnV TxV, 90/50, age 30',
    #             'TnV TxV, 70/30, age 30',

    #         ],
    #     filestem='_nov27'

    # )

    # plot_CEA_PDVAC(
    #     locations=locations,
    #     background_scens={

    #             '90-0-0': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
    #             # '90-35-70': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 35% sc cov'},
    #             '90-70-90': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 70% sc cov, 90% tx cov'},
    #         },

    #     txvx_scens=[
    #             'Mass TxV, 90/50, age 30',
    #             # 'Mass TxV, 70/30, age 30',
    #             # 'Mass TxV, 90/50, age 30, intro 2035',
    #             'TnV TxV, 90/50, age 30',
    #             # 'TnV TxV, 70/30, age 30',

    #         ],

    #     hpv_test_cost_range = [5,10,15,20],
    #     filestem='_nov27'

    #     )

    # plot_CEA_DEG(
    #     locations=locations,
    #     background_scens={
    #         # "90-0-0": {"vx_scen": "Vx, 90% cov, 9-14", "screen_scen": "No screening"},
    #         "90-35-70": {
    #             "vx_scen": "Vx, 90% cov, 9-14",
    #             "screen_scen": "HPV, 35% sc cov",
    #         },
    #         "90-70-90": {
    #             "vx_scen": "Vx, 90% cov, 9-14",
    #             "screen_scen": "HPV, 70% sc cov, 90% tx cov",
    #         },
    #     },
    #     txvx_scens=[
    #         "Mass TxV, 90/50, age 30",
    #         # 'Mass TxV, 70/30, age 30',
    #         # 'Mass TxV, 90/50, age 30, intro 2035',
    #         "TnV TxV, 90/50, age 30",
    #         # 'TnV TxV, 70/30, age 30',
    #     ],
    #     hpv_test_cost_range=[5, 10, 15, 20],
    #     filestem="_nov27",
    # )

    # plot_CEA_SnT(
    #     locations=locations,
    #     background_scens={
    #         "70-0-0": {"vx_scen": "Vx, 70% cov, 9-14", "screen_scen": "No screening"},
    #         "70-35-70": {
    #             "vx_scen": "Vx, 70% cov, 9-14",
    #             "screen_scen": "HPV, 35% sc cov",
    #         },
    #         "70-70-90": {
    #             "vx_scen": "Vx, 70% cov, 9-14",
    #             "screen_scen": "HPV, 70% sc cov, 90% tx cov",
    #         },
    #     },
    #     txvx_scens=[
    #         "No TxV",
    #         # "Mass TxV, 90/50, age 30",
    #         # "TnV TxV, 90/50, age 30",
    #     ],
    #     filestem="_feb21",
    # )

    plot_residual_burden_combined(
        locations=locations,
        scens={
            # "50% PxV, 0% S&T": ["Vx, 50% cov, 9-14", "No screening"],
            "70% PxV, 0% S&T": ["Vx, 70% cov, 9-14", "No screening"],
            "70% PxV, 35% Sc, 50% LTFU": [
                "Vx, 70% cov, 9-14",
                "HPV, 35% sc cov, 50% LTFU",
            ],
            "70% PxV, 70% Sc, 50% LTFU": [
                "Vx, 70% cov, 9-14",
                "HPV, 70% sc cov, 50% LTFU",
            ],
        },
        filestem="_feb25",
    )
