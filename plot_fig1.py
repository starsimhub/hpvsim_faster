import sciris as sc
import pandas as pd
import pylab as pl
import numpy as np
import utils as ut
from matplotlib.gridspec import GridSpec


def plot_fig1(locations, scens, filestem):

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
    fig_name = f"{ut.figfolder}/Fig1.png"
    sc.savefig(fig_name, dpi=100)

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

    plot_fig1(
        locations=locations,
        scens={
            "Scenario 1": ["Vx, 70% cov, 9-14", "No screening"],
            "Scenario 2": [
                "Vx, 70% cov, 9-14",
                "HPV, 35% sc cov, 50% LTFU",
            ],
            "Scenario 3": [
                "Vx, 70% cov, 9-14",
                "HPV, 70% sc cov, 50% LTFU",
            ],
            "Scenario 4": [
                "Vx, 70% cov, 9-14",
                "HPV, 70% sc cov, 30% LTFU",
            ],
        },
        filestem="_feb25",
    )

    T.toc("Done")
