import sciris as sc
import pandas as pd
import pylab as pl
import numpy as np
import utils as ut
from matplotlib.gridspec import GridSpec

def plot_fig1(locations, scens):

    ut.set_font(size=24)
    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{ut.resfolder}/{location}.obj')
        dfs += df
    bigdf = pd.concat(dfs)
    colors = ['gold', 'orange', 'red', 'darkred']

    fig = pl.figure(constrained_layout=True, figsize=(16, 20))
    spec2 = GridSpec(ncols=3, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec2[0, :])
    axes = []
    for i in np.arange(1,4):
        for j in np.arange(0,3):
            ax = fig.add_subplot(spec2[i,j])
            axes.append(ax)
    txvx_scen_label = 'No TxV'
    for ib, (ib_label, ib_scens) in enumerate(scens.items()):
        vx_scen_label = ib_scens[0]
        screen_scen_label = ib_scens[1]
        df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                   & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                 ['cancers', 'cancers_low', 'cancers_high']].sum()[2020:]
        years = np.array(df.index)
        ax1.plot(years, df['cancers'], color=colors[ib], label=ib_label)
        ax1.fill_between(years, df['cancers_low'], df['cancers_high'], color=colors[ib], alpha=0.3)
        ax1.set_ylabel('Cervical cancer cases')

    for pn, location in enumerate(locations):
        ax = axes[pn]
        for ib, (ib_label, ib_scens) in enumerate(scens.items()):
            vx_scen_label = ib_scens[0]
            screen_scen_label = ib_scens[1]
            df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                       & (bigdf.txvx_scen == txvx_scen_label) & (bigdf.location == location)].groupby('year')[
                     ['cancers', 'cancers_low', 'cancers_high']].sum()[2020:]
            years = np.array(df.index)
            ax.plot(years, df['cancers'], color=colors[ib])
            ax.fill_between(years, df['cancers_low'], df['cancers_high'], color=colors[ib], alpha=0.3)
            ax.set_ylabel('Cervical cancer cases')
            sc.SIticks(ax)
            if location == 'drc':
                ax.set_title('DRC')
            else:
                ax.set_title(location.capitalize())
    for pn in range(len(locations)):
        axes[pn].set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    ax1.set_title('Cumulative residual burden in all 9 countries')
    sc.SIticks(ax1)
    ax1.legend()
    fig.tight_layout()
    fig_name = f'{ut.figfolder}/Fig1.png'
    sc.savefig(fig_name, dpi=100)

    return

def plot_fig1_by_country(locations, scens, txv_scens):

    ut.set_font(size=24)
    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{ut.resfolder}/{location}.obj')
        dfs += df
    bigdf = pd.concat(dfs)
    colors = ['gold', 'orange', 'red', 'darkred']

    fig = pl.figure(constrained_layout=True, figsize=(16, 20))
    spec2 = GridSpec(ncols=3, nrows=4, figure=fig)
    ax1 = fig.add_subplot(spec2[0, :])
    axes = []
    for i in np.arange(1,4):
        for j in np.arange(0,3):
            ax = fig.add_subplot(spec2[i,j])
            axes.append(ax)
    for it, (txvx_scen, txvx_scen_label) in enumerate(txv_scens.items()):
        
        for ib, (ib_label, ib_scens) in enumerate(scens.items()):
            vx_scen_label = ib_scens[0]
            screen_scen_label = ib_scens[1]
            df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                    & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high']].sum()[2020:]
            years = np.array(df.index)
            if it == 0:
                ax1.plot(years, df['cancers'], color=colors[ib], label=ib_label)
            else:
                ax1.plot(years, df['cancers'], color=colors[ib], linestyle=':', label=ib_label)
            ax1.fill_between(years, df['cancers_low'], df['cancers_high'], color=colors[ib], alpha=0.3)
            ax1.set_ylabel('Cervical cancer cases')

    for pn, location in enumerate(locations):
        ax = axes[pn]
        for it, (txvx_scen, txvx_scen_label) in enumerate(txv_scens.items()):
            for ib, (ib_label, ib_scens) in enumerate(scens.items()):
                vx_scen_label = ib_scens[0]
                screen_scen_label = ib_scens[1]
                df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                        & (bigdf.txvx_scen == txvx_scen_label) & (bigdf.location == location)].groupby('year')[
                        ['cancers', 'cancers_low', 'cancers_high']].sum()[2020:]
                years = np.array(df.index)
                if it == 0:
                    ax.plot(years, df['cancers'], color=colors[ib])
                else:
                    ax.plot(years, df['cancers'], color=colors[ib], linestyle=':')
                ax.fill_between(years, df['cancers_low'], df['cancers_high'], color=colors[ib], alpha=0.3)
                ax.set_ylabel('Cervical cancer cases')
                sc.SIticks(ax)
                if location == 'drc':
                    ax.set_title('DRC')
                else:
                    ax.set_title(location.capitalize())
    for pn in range(len(locations)):
        axes[pn].set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    ax1.set_title('Cumulative residual burden in all 9 countries')
    sc.SIticks(ax1)
    ax1.legend()
    fig.tight_layout()
    fig_name = f'{ut.figfolder}/Fig1.png'
    sc.savefig(fig_name, dpi=100)

    return

def plot_fig1_cumulative(locations, scens, txv_scens):

    ut.set_font(size=24)
    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{ut.resfolder}/{location}.obj')
        dfs += df
        df = sc.loadobj(f'{ut.resfolder}/{location}_txv_cov.obj')
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
            df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                    & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high']].sum()[2030:]
            
            years = np.array(df.index)
            cum_cases =np.sum(df['cancers'])

            if txvx_scen_label == 'No TxV':
                txvx_scen_label_to_plot = txvx_scen_label
            else:
                txvx_scen_label_to_plot = txvx_scen
            if ib==0:
                ax.bar(xes[it][ib], cum_cases, width, color=colors[it], edgecolor='black', label=txvx_scen_label_to_plot)
            else:
                ax.bar(xes[it][ib], cum_cases, width, color=colors[it], edgecolor='black')

            ax.text(xes[it][ib], cum_cases+10e4, round(cum_cases/1e6, 1), ha='center')
            
    ax.set_ylabel('Cervical cancer cases (2030-2060)')
    ax.set_xticks(x + width, scens.keys())
    ax.set_xlabel('Background intervention scenario (PxV-Sc-Tx)')
    ax.set_ylim(top=12.5e6)
    sc.SIticks(ax)
    ax.legend()#ncol=2)
        
    fig.tight_layout()
    fig_name = f'{ut.figfolder}/CC_burden.png'
    sc.savefig(fig_name, dpi=100)

    return

def plot_txv_efficacies(locations=None, background_scens=None, txvx_age=None, txvx_efficacies=None):


    ut.set_font(size=24)

    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{ut.resfolder}/{location}.obj')
        dfs += df
    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(20)
    width = 0.2

    r1 = np.arange(len(background_scens))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    r4 = [x + width for x in r3]
    xes = [r1, r2, r3, r4]

    fig, ax = pl.subplots(figsize=(16, 8))

    
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        no_txv_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                    & (bigdf.txvx_scen == 'No TxV')].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high']].sum()[2020:]
        cum_cases_no_txv =np.sum(no_txv_df['cancers']) 
        for i_eff, txvx_efficacy in enumerate(txvx_efficacies):
            txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'
            txv_df = bigdf[(bigdf.screen_scen == screen_scen_label) & (bigdf.vx_scen == vx_scen_label)
                    & (bigdf.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high']].sum()[2020:]
            cum_cases_txv =np.sum(txv_df['cancers'])
            cum_cases_averted = cum_cases_no_txv - cum_cases_txv

            if ib == 0:
                ax.bar(xes[i_eff][ib], cum_cases_averted, width=0.2, color=colors[i_eff], label=txvx_efficacy)
            else:
                ax.bar(xes[i_eff][ib], cum_cases_averted, width=0.2, color=colors[i_eff])
     
    ib_labels = background_scens.keys()
    ax.set_xticks([r + 1.5*width for r in range(len(r2))], ib_labels)
    sc.SIticks(ax)
    ax.legend(title='TxV effectiveness\n(HPV/CIN2+)')


    ax.set_ylabel(f'Cervical cancer cases averted\n (2030-2060)')

    ax.set_xlabel('Background intervention scenario')
    fig.tight_layout()
    fig_name = f'{ut.figfolder}/TxV_efficacy.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_burden(locations, vx_scen, screen_scen, txvx_scen):
    ut.set_font(size=24)

    dfs = sc.autolist()
    for location in locations:
        df = sc.loadobj(f'{ut.resfolder}/{location}_delayed_intro.obj')
        dfs += df
    bigdf = pd.concat(dfs)
    colors = sc.gridcolors(20)
    
    fig, ax = pl.subplots(figsize=(16, 8))
    df = bigdf[(bigdf.screen_scen == screen_scen) & (bigdf.vx_scen == vx_scen) & (bigdf.txvx_scen == txvx_scen)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high']].sum()[2020:]
    years = np.array(df.index)
    ax.plot(years, df['cancers'], label=location.capitalize())
    ax.fill_between(years, df['cancers_low'], df['cancers_high'], alpha=0.3)
    fig.tight_layout()
    fig_name = f'{ut.figfolder}/delayed_intro.png'
    fig.savefig(fig_name, dpi=100)
    

# %% Run as a script
if __name__ == '__main__':
    T = sc.timer()

    locations = [
        'india',  # 0
        'indonesia',  # 1
        'nigeria',  # 2
        'tanzania',  # 3
        'bangladesh',  # 4
        'myanmar',  # 5
        'uganda',  # 6
        'ethiopia',  # 7
        'drc',  # 8
        # 'kenya'  # 9
    ]

    # plot_txv_efficacies(
    #     locations=locations,
    #     background_scens={
    #         '90-0-0': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
    #         '90-35-70': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 35% sc cov'},
    #         '90-70-90': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 70% sc cov, 90% tx cov'},
    #     },
    #     txvx_efficacies=['90/0', '70/30', '50/50', '90/50'],
    #     txvx_age='30',
    # )

    plot_fig1_cumulative(
        locations=locations,
        scens={
            '90-0-0': ['Vx, 90% cov, 9-14', 'No screening'],
            '90-35-70': ['Vx, 90% cov, 9-14', 'HPV, 35% sc cov'],
            '90-70-90': ['Vx, 90% cov, 9-14', 'HPV, 70% sc cov, 90% tx cov'],
        },
        txv_scens={
            'No TxV': 'No TxV',
            '90/50 TxV, 20% cov': 'Mass TxV, 90/50, age 30, 20 cov', 
            '90/50 TxV, 50% cov': 'Mass TxV, 90/50, age 30, 50 cov',
            '90/50 TxV, 70% cov': 'Mass TxV, 90/50, age 30',
        }
    )
    
    # plot_burden(
    #     locations=locations,
    #     vx_scen = 'Vx, 90% cov, 9-14', 
    #     screen_scen= 'No screening',
    #     txvx_scen='No TxV'
        
    # )


    T.toc('Done')