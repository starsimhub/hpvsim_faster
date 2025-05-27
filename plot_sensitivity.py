import sciris as sc
import pandas as pd
import pylab as pl
import numpy as np
import utils as ut
from matplotlib.gridspec import GridSpec

def plot_fig4(locations=None, discounting=False, background_scen=None, txvx_efficacy=None, txvx_ages=None, sensitivities=None):

    sens_labels = {
        '': ['Baseline', 'baseline'],
        ', cross-protection': ['50% cross-protection', '50%\ncross-protection'],
        ', 0.05 decay': ['5% annual decay since virus/lesion', '0.05decay'],
        ', intro 2035':['2035 introduction', '2035\nintroduction'],
        ', no durable immunity': ['No immune memory', 'No immune\nmemory'],
        '90/0': ['90% HPV clearance,\n0% CIN2+ clearance', '90% HPV\nclearance,\n0% CIN2+\nclearance'],
    }
    sens_labels_to_use = [sens_labels[sens_label][0] for sens_label in sensitivities]
    ut.set_font(size=20)
    econdfs = sc.autolist()
    
    delayed_intro_dfs = sc.autolist()

    for location in locations:
        econ_df = sc.loadobj(f'{ut.resfolder}/{location}.obj')
        econdfs += econ_df
        delayed_intro_df = sc.loadobj(f'{ut.resfolder}/{location}_delayed_intro.obj')
        delayed_intro_dfs += delayed_intro_df
        

    econdf = pd.concat(econdfs)
    delayed_df = pd.concat(delayed_intro_dfs)
    colors = sc.gridcolors(20)[4:]
    width = 0.25


    r1 = np.arange(len(sensitivities))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']
    fig, ax = pl.subplots(figsize=(14, 14))
    vx_scen_label = background_scen['vx_scen']
    screen_scen_label = background_scen['screen_scen']
    NoTxV = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                  & (econdf.txvx_scen == 'No TxV')].groupby('year')[
        ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',]].sum()

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV['cancers'])[ys:ye])

    for i_age, txvx_age in enumerate(txvx_ages):
        txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'

        TxV_baseline = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                     & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
             'cancer_deaths_high', ]].sum()
        TxV_cancers_baseline = np.sum(np.array(TxV_baseline['cancers'])[ys:ye])

        cancers_averted_baseline = NoTxV_cancers - TxV_cancers_baseline

        for isens, sens_label in enumerate(sensitivities):
            if sens_label == '90/0':
                txvx_scen_label_sen = f'Mass TxV, 90/0, age {txvx_age}'
            else:
                txvx_scen_label_sen = f'{txvx_scen_label}{sens_label}'

            

            if 'intro 2035' in sens_label:
                TxV = \
                delayed_df[(delayed_df.screen_scen == screen_scen_label) & (delayed_df.vx_scen == vx_scen_label)
                       & (delayed_df.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', ]].sum()
                ys = sc.findinds(TxV.index, 2035)[0]
                ye = sc.findinds(TxV.index, 2065)[0]
                TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
            else:
                TxV = \
                econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', ]].sum()
                ys = sc.findinds(TxV.index, 2030)[0]
                ye = sc.findinds(TxV.index, 2060)[0]
                TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
            cancers_averted = NoTxV_cancers - TxV_cancers
            print(f'{sens_label}, {txvx_age}: {cancers_averted - cancers_averted_baseline}')

            if isens == 0:
                ax.bar(xes[i_age][isens], cancers_averted - cancers_averted_baseline,
                           color=colors[i_age], width=width, edgecolor='black', label=txvx_age)
            else:
                ax.bar(xes[i_age][isens], cancers_averted - cancers_averted_baseline, width=width, edgecolor='black',
                           color=colors[i_age])

    sc.SIticks(ax)
    ax.axhline(0, color='black')
    # ax.set_ylim(-75,40)
    ax.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use)

    ax.set_ylabel(f'Cervical cancer cases averted\n(relative to baseline TxV*, see below)')

    ax.legend(title='Age of TxV')
    fig.tight_layout()

    fig_name = f'{ut.figfolder}/sensitivity.png'
    fig.savefig(fig_name, dpi=100)


    return


def plot_fig4_v2(locations=None, background_scen=None, txvx_efficacy=None, txvx_ages=None, sensitivities=None):

    sens_labels = {
        '': ['Baseline', 'baseline'],
        ', cross-protection': ['50% cross-protection', '50%\ncross-protection'],
        ', 0.05 decay': ['5% annual decay since virus/lesion', '0.05decay'],
        ', intro 2035':['2035 introduction', '2035\nintroduction'],
        ', no durable immunity': ['No immune memory', 'No immune\nmemory'],
        '90/0': ['90% HPV clearance,\n0% CIN2+ clearance', '90% HPV\nclearance,\n0% CIN2+\nclearance'],
        ', 50 cov': ['50% coverage', '50% \ncoverage']
    }
    sens_labels_to_use = [sens_labels[sens_label][0] for sens_label in sensitivities]
    ut.set_font(size=20)
    econdfs = sc.autolist()
    
    delayed_intro_dfs = sc.autolist()

    for location in locations:
        econ_df = sc.loadobj(f'{ut.resfolder}/{location}.obj')
        econdfs += econ_df
        delayed_intro_df = sc.loadobj(f'{ut.resfolder}/{location}_delayed_intro.obj')
        delayed_intro_dfs += delayed_intro_df
        

    econdf = pd.concat(econdfs)
    delayed_df = pd.concat(delayed_intro_dfs)
    colors = sc.gridcolors(20)[4:]
    width = 0.2


    r1 = np.arange(len(sensitivities))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    fig, ax = pl.subplots(figsize=(14, 8))
    vx_scen_label = background_scen['vx_scen']
    screen_scen_label = background_scen['screen_scen']
    NoTxV = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                  & (econdf.txvx_scen == 'No TxV')].groupby('year')[
        ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',]].sum()

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV['cancers'])[ys:ye])

    for i_age, txvx_age in enumerate(txvx_ages):
        txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'

        TxV_baseline = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                     & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
             'cancer_deaths_high', ]].sum()

        TxV_cancers_baseline = np.sum(np.array(TxV_baseline['cancers'])[ys:ye])
        if i_age == 0:
            ax.axhline(NoTxV_cancers-TxV_cancers_baseline, color=colors[i_age], linewidth = 2, label='Baseline TxV')
        else:
            ax.axhline(NoTxV_cancers-TxV_cancers_baseline, color=colors[i_age], linewidth = 2)

        for isens, sens_label in enumerate(sensitivities):
            if sens_label == '90/0':
                txvx_scen_label_sen = f'Mass TxV, 90/0, age {txvx_age}'
            else:
                txvx_scen_label_sen = f'{txvx_scen_label}{sens_label}'

            if 'intro 2035' in sens_label:
                TxV = \
                delayed_df[(delayed_df.screen_scen == screen_scen_label) & (delayed_df.vx_scen == vx_scen_label)
                       & (delayed_df.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', ]].sum()
                NoTxV_delayed = delayed_df[(delayed_df.screen_scen == screen_scen_label) & (delayed_df.vx_scen == vx_scen_label)
                                  & (delayed_df.txvx_scen == 'No TxV')].groupby('year')[['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',]].sum()

                ys = sc.findinds(TxV.index, 2035)[0]
                ye = sc.findinds(TxV.index, 2065)[0]
                TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
                no_TxV_cancers = np.sum(np.array(NoTxV_delayed['cancers'])[ys:ye])
                averted_cancers = no_TxV_cancers - TxV_cancers
            else:
                TxV = \
                econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', ]].sum()
                ys = sc.findinds(TxV.index, 2030)[0]
                ye = sc.findinds(TxV.index, 2060)[0]
                TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
                averted_cancers = NoTxV_cancers - TxV_cancers
            
            if isens == 0:
                ax.bar(xes[i_age][isens], averted_cancers,
                           color=colors[i_age], width=width, edgecolor='black', label=txvx_age)
                # ax.scatter(xes[i_age][isens], averted_cancers,
                #            color=colors[i_age], s=400, edgecolor='black', label=txvx_age)

            else:
                ax.bar(xes[i_age][isens], averted_cancers, width=width, edgecolor='black',
                           color=colors[i_age])
                # ax.scatter(xes[i_age][isens], averted_cancers, s=400, edgecolor='black',
                #            color=colors[i_age])

    sc.SIticks(ax)
    ax.set_ylim(9e5,3e6)
    # ax.set_ylim(9e5,2.55e6)
    ax.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use)

    ax.set_ylabel(f'Cervical cancer cases averted')

    ax.legend(title='Age of TxV')
    fig.tight_layout()

    fig_name = f'{ut.figfolder}/sensitivity_v2.png'
    fig.savefig(fig_name, dpi=100)


    return


def plot_sens(locations=None, background_scen=None, txvx_efficacy=None, txvx_ages=None, sensitivities=None):

    sens_labels = {
        '': ['Baseline', 'baseline'],
        ', cross-protection': ['50% cross-protection', '50%\ncross-protection'],
        ', 0.05 decay': ['5% annual decay since virus/lesion', '0.05decay'],
        ', intro 2035':['2035 introduction', '2035\nintroduction'],
        ', no durable immunity': ['No immune memory', 'No immune\nmemory'],
        '90/0': ['90% HPV clearance,\n0% CIN2+ clearance', '90% HPV\nclearance,\n0% CIN2+\nclearance'],
        ', 50 cov': ['50% coverage', '50% \ncoverage']
    }
    sens_labels_to_use = [sens_labels[sens_label][0] for sens_label in sensitivities]
    ut.set_font(size=20)
    econdfs = sc.autolist()
    
    delayed_intro_dfs = sc.autolist()

    for location in locations:
        econ_df = sc.loadobj(f'{ut.resfolder}/{location}_nov25.obj')
        econdfs += econ_df
        delayed_intro_df = sc.loadobj(f'{ut.resfolder}/{location}_delayed_intro.obj')
        delayed_intro_dfs += delayed_intro_df
        

    econdf = pd.concat(econdfs)
    delayed_df = pd.concat(delayed_intro_dfs)
    colors = sc.gridcolors(20)[4:]
    width = 0.2


    r1 = np.arange(len(sensitivities))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    fig, ax = pl.subplots(figsize=(14, 8))
    vx_scen_label = background_scen['vx_scen']
    screen_scen_label = background_scen['screen_scen']
    NoTxV = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                                  & (econdf.txvx_scen == 'No TxV')].groupby('year')[
        ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',]].sum()

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV['cancers'])[ys:ye])

    for i_age, txvx_age in enumerate(txvx_ages):
        txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'

        TxV_baseline = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                     & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
            ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
             'cancer_deaths_high', ]].sum()

        TxV_cancers_baseline = np.sum(np.array(TxV_baseline['cancers'])[ys:ye])
        if i_age == 0:
            ax.axhline(NoTxV_cancers-TxV_cancers_baseline, color=colors[i_age], linewidth = 2, label='Baseline TxV')
        else:
            ax.axhline(NoTxV_cancers-TxV_cancers_baseline, color=colors[i_age], linewidth = 2)

        for isens, sens_label in enumerate(sensitivities):
            if sens_label == '90/0':
                txvx_scen_label_sen = f'Mass TxV, 90/0, age {txvx_age}'
            else:
                txvx_scen_label_sen = f'{txvx_scen_label}{sens_label}'

            if 'intro 2035' in sens_label:
                TxV = \
                delayed_df[(delayed_df.screen_scen == screen_scen_label) & (delayed_df.vx_scen == vx_scen_label)
                       & (delayed_df.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', ]].sum()
                NoTxV_delayed = delayed_df[(delayed_df.screen_scen == screen_scen_label) & (delayed_df.vx_scen == vx_scen_label)
                                  & (delayed_df.txvx_scen == 'No TxV')].groupby('year')[['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',]].sum()

                ys = sc.findinds(TxV.index, 2035)[0]
                ye = sc.findinds(TxV.index, 2065)[0]
                TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
                no_TxV_cancers = np.sum(np.array(NoTxV_delayed['cancers'])[ys:ye])
                averted_cancers = no_TxV_cancers - TxV_cancers
            else:
                TxV = \
                econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label_sen)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', ]].sum()
                ys = sc.findinds(TxV.index, 2030)[0]
                ye = sc.findinds(TxV.index, 2060)[0]
                TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
                averted_cancers = NoTxV_cancers - TxV_cancers
            
            if isens == 0:
                ax.bar(xes[i_age][isens], averted_cancers,
                           color=colors[i_age], width=width, edgecolor='black', label=txvx_age)
                # ax.scatter(xes[i_age][isens], averted_cancers,
                #            color=colors[i_age], s=400, edgecolor='black', label=txvx_age)

            else:
                ax.bar(xes[i_age][isens], averted_cancers, width=width, edgecolor='black',
                           color=colors[i_age])
                # ax.scatter(xes[i_age][isens], averted_cancers, s=400, edgecolor='black',
                #            color=colors[i_age])

    sc.SIticks(ax)
    ax.set_ylim(9e5,3e6)
    # ax.set_ylim(9e5,2.55e6)
    ax.set_xticks([r + width for r in range(len(r1))], sens_labels_to_use)

    ax.set_ylabel(f'Cervical cancer cases averted')

    ax.legend(title='Age of TxV')
    fig.tight_layout()

    fig_name = f'{ut.figfolder}/sensitivity.png'
    fig.savefig(fig_name, dpi=100)


def plot_tnv_sens(locations=None, background_scen=None, txvx_efficacies=None, txvx_age=None):

    ut.set_font(size=20)
    dfs = sc.autolist()
    econdfs = sc.autolist()

    for location in locations:
        df = sc.loadobj(f'{ut.resfolder}/{location}.obj')
        dfs += df
        df = sc.loadobj(f'{ut.resfolder}/{location}_tnv.obj')
        dfs += df
        
        econdf = sc.loadobj(f'{ut.resfolder}/{location}_econ.obj')
        econdfs += econdf
        econdf = sc.loadobj(f'{ut.resfolder}/{location}_tnv_econ.obj')
        econdfs += econdf
        
    df = pd.concat(dfs)
    econdf = pd.concat(econdfs)
    colors = sc.gridcolors(20)[7:]
    width = 0.25

    r1 = np.arange(len(txvx_efficacies))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    fig, ax = pl.subplots(figsize=(14, 8))
    vx_scen_label = background_scen['vx_scen']
    screen_scen_label = background_scen['screen_scen']
    NoTxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                                  & (df.txvx_scen == 'No TxV')].groupby('year')[
        ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',]].sum()

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV['cancers'])[ys:ye])

    for itxv, txvx_efficacy in enumerate(txvx_efficacies):
        
        txvx_scen_label = f'TnV TxV, {txvx_efficacy}, age {txvx_age}'
        TxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                       & (df.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', 'n_tx_vaccinated', 'n_screened' ]].sum()
        ys = sc.findinds(TxV.index, 2030)[0]
        ye = sc.findinds(TxV.index, 2060)[0]
        TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
        averted_cancers = NoTxV_cancers - TxV_cancers
        
        TxV_econ = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['new_tx_vaccinations', 'new_hpv_screens']].sum()
        ys = sc.findinds(TxV_econ.index, 2030)[0]
        ye = sc.findinds(TxV_econ.index, 2060)[0]
        TxV_vaccinations = np.sum(np.array(TxV_econ['new_tx_vaccinations'])[ys:ye])
            
        mass_txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'
        mass_TxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                       & (df.txvx_scen == mass_txvx_scen_label)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', 'n_tx_vaccinated', 'n_screened' ]].sum()
        ys = sc.findinds(TxV.index, 2030)[0]
        ye = sc.findinds(TxV.index, 2060)[0]
        mass_TxV_cancers = np.sum(np.array(mass_TxV['cancers'])[ys:ye])
        mass_averted_cancers = NoTxV_cancers - mass_TxV_cancers
        
        mass_TxV_econ = econdf[(econdf.screen_scen == screen_scen_label) & (econdf.vx_scen == vx_scen_label)
                       & (econdf.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['new_tx_vaccinations']].sum()
        ys = sc.findinds(mass_TxV_econ.index, 2030)[0]
        ye = sc.findinds(mass_TxV_econ.index, 2060)[0]
        mass_TxV_vaccinations = np.sum(np.array(mass_TxV_econ['new_tx_vaccinations'])[ys:ye])
        # print(f'{txvx_efficacy}: {np.array(TxV["n_tx_vaccinated"])[-1]} TnV, {np.array(mass_TxV["n_tx_vaccinated"])[-1]} Mass Vx')
        if itxv == 0:
            ax.bar(xes[0][itxv], averted_cancers,
                        color=colors[0], width=width, edgecolor='black', label='Test and vaccinate')
            ax.bar(xes[1][itxv], mass_averted_cancers,
                        color=colors[1], width=width, edgecolor='black', label='Mass vaccinate')
                
                
        else:
            ax.bar(xes[0][itxv], averted_cancers, width=width, edgecolor='black',
                           color=colors[0])
            ax.bar(xes[1][itxv], mass_averted_cancers,
                        color=colors[1], width=width, edgecolor='black') 
            

          
    print(f'{round(np.array(TxV["n_tx_vaccinated"])[-1]/1e6,0)} million TxV (TnV), {round(np.array(TxV["n_screened"])[-1]/1e6,0)} million HPV tests (TnV), {round(np.array(mass_TxV["n_tx_vaccinated"])[-1]/1e6,0)} million TxV (Mass Vx)')
    sc.SIticks(ax)
    ax.set_xticks([r + width/2 for r in range(len(r1))], txvx_efficacies)

    ax.set_ylabel(f'Cervical cancer cases averted')
    ax.set_xlabel(f'TxV % Effectiveness (HPV/CIN2+ clearance)')

    ax.legend(title='TxV Delivery (age 30-40)')
    fig.tight_layout()

    fig_name = f'{ut.figfolder}/tnv_sens.png'
    fig.savefig(fig_name, dpi=100)
    
    fig, ax = pl.subplots(figsize=(14, 8))
    tvec = np.arange(2030, 2060)
    ys = sc.findinds(mass_TxV.index, 2030)[0]
    ye = sc.findinds(mass_TxV.index, 2060)[0]
    ax.plot(tvec, np.array(mass_TxV["n_tx_vaccinated"])[ys:ye], color=colors[0], label='Mass Vx, n_tx_vaccinated')
    ax.plot(tvec, np.array(TxV["n_tx_vaccinated"])[ys:ye], color=colors[1], label='TnV Vx, n_tx_vaccinated')
    ax.plot(tvec, np.array(TxV["n_screened"])[ys:ye], color=colors[2], label='TnV Vx, n_screened')

    ys = sc.findinds(mass_TxV_econ.index, 2030)[0]
    ye = sc.findinds(mass_TxV_econ.index, 2060)[0]
    ax.plot(tvec, np.array(mass_TxV_econ['new_tx_vaccinations'])[ys:ye], color=colors[3], label='Mass Vx, new_tx_vaccinated')
    ax.plot(tvec, np.array(TxV_econ["new_tx_vaccinations"])[ys:ye], color=colors[4], label='TnV Vx, new_tx_vaccinated')
    ax.plot(tvec, np.array(TxV_econ["new_hpv_screens"])[ys:ye], color=colors[5], label='TnV Vx, new_screened')
    ax.legend()
    sc.SIticks(ax)
    fig.tight_layout()

    fig_name = f'{ut.figfolder}/n_txv.png'
    fig.savefig(fig_name, dpi=100)

    return

def plot_tnv_sens_v2(locations=None, background_scen=None, txvx_efficacies=None, txvx_age=None):

    ut.set_font(size=20)
    dfs = sc.autolist()

    for location in locations:
        df = sc.loadobj(f'{ut.resfolder}/{location}_nov25.obj')
        dfs += df
                
    df = pd.concat(dfs)
    colors = sc.gridcolors(20)[7:]
    width = 0.25

    r1 = np.arange(len(txvx_efficacies))
    r2 = [x + width for x in r1]
    r3 = [x + width for x in r2]
    xes = [r1, r2, r3]
    fig, ax = pl.subplots(figsize=(14, 8))
    vx_scen_label = background_scen['vx_scen']
    screen_scen_label = background_scen['screen_scen']
    NoTxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                                  & (df.txvx_scen == 'No TxV')].groupby('year')[
        ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low', 'cancer_deaths_high',]].sum()

    ys = sc.findinds(NoTxV.index, 2030)[0]
    ye = sc.findinds(NoTxV.index, 2060)[0]

    NoTxV_cancers = np.sum(np.array(NoTxV['cancers'])[ys:ye])

    for itxv, txvx_efficacy in enumerate(txvx_efficacies):
        
        txvx_scen_label = f'TnV TxV, {txvx_efficacy}, age {txvx_age}'
        TxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                       & (df.txvx_scen == txvx_scen_label)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', 'new_txvx_doses', 'new_screens' ]].sum()
        ys = sc.findinds(TxV.index, 2030)[0]
        ye = sc.findinds(TxV.index, 2060)[0]
        TxV_cancers = np.sum(np.array(TxV['cancers'])[ys:ye])
        averted_cancers = NoTxV_cancers - TxV_cancers
        TxV_vaccinations = np.sum(np.array(TxV['new_txvx_doses'])[ys:ye])
            
        mass_txvx_scen_label = f'Mass TxV, {txvx_efficacy}, age {txvx_age}'
        mass_TxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                       & (df.txvx_scen == mass_txvx_scen_label)].groupby('year')[
                    ['cancers', 'cancers_low', 'cancers_high', 'cancer_deaths', 'cancer_deaths_low',
                     'cancer_deaths_high', 'new_txvx_doses', 'new_screens' ]].sum()
        ys = sc.findinds(TxV.index, 2030)[0]
        ye = sc.findinds(TxV.index, 2060)[0]
        mass_TxV_cancers = np.sum(np.array(mass_TxV['cancers'])[ys:ye])
        mass_averted_cancers = NoTxV_cancers - mass_TxV_cancers
        
        mass_TxV_vaccinations = np.sum(np.array(mass_TxV['new_txvx_doses'])[ys:ye])
        
        if itxv == 0:
            ax.bar(xes[0][itxv], averted_cancers,
                        color=colors[0], width=width, edgecolor='black', label='Test and vaccinate')
            ax.bar(xes[1][itxv], mass_averted_cancers,
                        color=colors[1], width=width, edgecolor='black', label='Mass vaccinate')
                
                
        else:
            ax.bar(xes[0][itxv], averted_cancers, width=width, edgecolor='black',
                           color=colors[0])
            ax.bar(xes[1][itxv], mass_averted_cancers,
                        color=colors[1], width=width, edgecolor='black') 
            

          
    print(f'{round(TxV_vaccinations/1e6,0)} million TxV (TnV), {round(mass_TxV_vaccinations/1e6,0)} million TxV (Mass Vx)')
    sc.SIticks(ax)
    ax.set_xticks([r + width/2 for r in range(len(r1))], txvx_efficacies)

    ax.set_ylabel(f'Cervical cancer cases averted')
    ax.set_xlabel(f'TxV % Effectiveness (HPV/CIN2+ clearance)')

    ax.legend(title='TxV Delivery (age 30-40)')
    fig.tight_layout()

    fig_name = f'{ut.figfolder}/tnv_sens_v2.png'
    fig.savefig(fig_name, dpi=100)
    
    # fig, ax = pl.subplots(figsize=(14, 8))
    # tvec = np.arange(2030, 2060)
    # ys = sc.findinds(mass_TxV.index, 2030)[0]
    # ye = sc.findinds(mass_TxV.index, 2060)[0]
    # ax.plot(tvec, np.array(mass_TxV["n_tx_vaccinated"])[ys:ye], color=colors[0], label='Mass Vx, n_tx_vaccinated')
    # ax.plot(tvec, np.array(TxV["n_tx_vaccinated"])[ys:ye], color=colors[1], label='TnV Vx, n_tx_vaccinated')
    # ax.plot(tvec, np.array(TxV["n_screened"])[ys:ye], color=colors[2], label='TnV Vx, n_screened')

    # ys = sc.findinds(mass_TxV_econ.index, 2030)[0]
    # ye = sc.findinds(mass_TxV_econ.index, 2060)[0]
    # ax.plot(tvec, np.array(mass_TxV_econ['new_tx_vaccinations'])[ys:ye], color=colors[3], label='Mass Vx, new_tx_vaccinated')
    # ax.plot(tvec, np.array(TxV_econ["new_tx_vaccinations"])[ys:ye], color=colors[4], label='TnV Vx, new_tx_vaccinated')
    # ax.plot(tvec, np.array(TxV_econ["new_hpv_screens"])[ys:ye], color=colors[5], label='TnV Vx, new_screened')
    # ax.legend()
    # sc.SIticks(ax)
    # fig.tight_layout()

    # fig_name = f'{ut.figfolder}/n_txv_v2.png'
    # fig.savefig(fig_name, dpi=100)

    return

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

    # plot_fig4_v2(
    #     locations=locations,
    #     background_scen={'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
    #     txvx_efficacy='90/50',
    #     txvx_ages=['30', '35', '40'],
    #     sensitivities=[', cross-protection', 
    #                    ', intro 2035', 
    #                    ', no durable immunity',
    #                    '90/0',
    #                    ]
    # )
    

    
    plot_tnv_sens_v2(
        locations=locations,
        background_scen={'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
        txvx_age='30',
        txvx_efficacies=['90/50', 
                       '50/50',
                       '70/30',
                       '90/0', 
                       ]
    )
    
    plot_sens(
        locations=locations,
        background_scen={'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
        txvx_efficacy='90/50',
        txvx_ages=['30', '35', '40'],
        sensitivities=[', cross-protection', 
                       ', intro 2035', 
                       ', no durable immunity',
                       '90/0',
                       ]
    )


    T.toc('Done')