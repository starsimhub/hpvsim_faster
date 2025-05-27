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
        vx_scen_label = ib_scens[0]
        screen_scen_label = ib_scens[1]
        for it, (txvx_scen, txvx_scen_label) in enumerate(txv_scens.items()):
            if ib == 0 and "HPV" in txvx_scen_label:
                pass
            elif (
                "HPV" in txvx_scen_label
                and txvx_scen_label[:-10] != screen_scen_label[:-10]
            ):
                pass
            else:
                if "HPV" in txvx_scen_label:
                    screen_scen_label_to_use = sc.dcp(txvx_scen_label)
                    txvx_scen_to_use = "No TxV"
                    it = 3
                else:
                    screen_scen_label_to_use = screen_scen_label
                    txvx_scen_to_use = txvx_scen_label

                df = (
                    bigdf[
                        (bigdf.screen_scen == screen_scen_label_to_use)
                        & (bigdf.vx_scen == vx_scen_label)
                        & (bigdf.txvx_scen == txvx_scen_to_use)
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

                cum_cases = np.sum(df["cancers"])
                cum_cases_2040 = np.sum(df["cancers"].values[:10])

                if txvx_scen_label == "No TxV":
                    txvx_scen_label_to_plot = "Option 1: Do nothing"
                else:
                    txvx_scen_label_to_plot = txvx_scen
                if ib == 1:
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
                        xes[it][ib],
                        cum_cases,
                        width,
                        color=colors[it],
                        edgecolor="black",
                    )
                ax.bar(
                    xes[it][ib],
                    cum_cases_2040,
                    width,
                    color=colors[it],
                    hatch="///",
                    edgecolor="black",
                )

                ax.text(
                    xes[it][ib],
                    cum_cases_2040 + 10e4,
                    round(cum_cases_2040 / 1e6, 1),
                    ha="center",
                )
                ax.text(
                    xes[it][ib],
                    cum_cases + 10e4,
                    round(cum_cases / 1e6, 1),
                    ha="center",
                )

    ax.set_ylabel("Cervical cancer cases (2030-2060)")
    ax.set_xticks(x + 1.5 * width, scens.keys())
    ax.set_xlabel("Background screen coverage")
    ax.set_ylim(top=15e6)
    sc.SIticks(ax)
    ax.legend(ncol=2)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/CC_burden_v2{filestem}.png"
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
                        np.sum(TxV["new_tx_vaccinations"] * cost_dict["txv"])
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

    fig, axes = pl.subplots(nrows=2, figsize=(10, 10), sharex=True)
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
                total_dalys = 0
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
                        np.sum(TxV["new_tx_vaccinations"] * cost_dict["txv"])
                        + np.sum(TxV["new_hpv_screens"] * cost_dict["hpv"])
                        + np.sum(TxV["new_poc_hpv_screens"] * cost_dict["poc_hpv"])
                        + np.sum(TxV["new_leeps"] * cost_dict["leep"])
                        + np.sum(TxV["new_thermal_ablations"] * cost_dict["ablation"])
                        + np.sum(TxV["new_cancer_treatments"] * cost_dict["cancer"])
                    )
                    total_costs += cost_TxV
                    total_dalys += TxV["dalys"]
                if ib == 2:

                    axes[0].bar(
                        xes[it][ib],
                        total_costs,
                        color=colors[it],
                        width=width,
                        edgecolor="black",
                        label=txvx_scen_label[txvx_scen],
                    )
                else:
                    axes[0].bar(
                        xes[it][ib],
                        total_costs,
                        color=colors[it],
                        width=width,
                        edgecolor="black",
                    )

                axes[0].text(
                    xes[it][ib],
                    total_costs + 100e6,
                    f"${round(total_costs / 1e9, 1)}",
                    ha="center",
                )

                axes[1].bar(
                    xes[it][ib],
                    total_dalys,
                    color=colors[it],
                    width=width,
                    edgecolor="black",
                )

                axes[1].text(
                    xes[it][ib],
                    total_dalys + 4e6,
                    f"{round(total_dalys[0] / 1e6, 1)}",
                    ha="center",
                )

    axes[1].set_xticks(x + 1.5 * width, background_scens.keys())

    # ax.set_ylim(top=15e9)
    axes[0].legend(ncol=2)
    axes[1].set_xlabel("Screen coverage")
    axes[0].set_ylabel("Total costs (2030-2060)")
    axes[1].set_ylabel("Total DALYs (2030-2060)")
    for ax in axes:
        sc.SIticks(ax)

    fig.tight_layout()
    fig_name = f"{ut.figfolder}/total_cost_dalys{filestem}.png"
    fig.savefig(fig_name, dpi=100)
    return


def plot_CEA(
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
                        np.sum(TxV["new_tx_vaccinations"] * cost_dict["txv"])
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

    plot_CEA(
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
            "HPV, 35% sc cov, 30% LTFU",
        ],
        filestem="_feb25",
    )

    plot_fig1(
        locations=locations,
        scens={
            "None": ["Vx, 70% cov, 9-14", "No screening"],
            "35%": [
                "Vx, 70% cov, 9-14",
                "HPV, 35% sc cov, 50% LTFU",
            ],
            "70%": [
                "Vx, 70% cov, 9-14",
                "HPV, 70% sc cov, 50% LTFU",
            ],
        },
        txv_scens={
            "Option 1: Do nothing": "No TxV",
            "Option 3a: Mass TxV (90/50)": "Mass TxV, 90/50, age 30",
            "Option 3b: TnV TxV (90/50)": "TnV TxV, 90/50, age 30",
            "Option 2a: POC HPV test (30% LTFU)": "HPV, 35% sc cov, 30% LTFU",
            "POC HPV test (30% LTFU)v2": "HPV, 70% sc cov, 30% LTFU",
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
