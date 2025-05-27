import sciris as sc
import pandas as pd
import pylab as pl
import numpy as np
import utils as ut
from matplotlib.gridspec import GridSpec

def plot_CEA(locations=None, background_scens=None, txvx_scens=None, discounting=False):

    ut.set_font(size=14)
    econdfs = sc.autolist()
    delayed_intro_dfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f'{ut.resfolder}/{location}_econ.obj')
        delayed_introdf = sc.loadobj(f'{ut.resfolder}/{location}_econ_delayed_intro.obj')
        delayed_intro_dfs += delayed_introdf
        econdfs += econdf
    econ_df = pd.concat(econdfs)
    delayed_intro_df = pd.concat(delayed_intro_dfs)

    cost_dict = dict(
        txv=8,
        leep=41.76,
        ablation=11.76,
        cancer=450
    )

    standard_le = 88.8
    colors = ['orange', 'red', 'darkred']
    markers = ['s', 'v', 'P', '*', '+', 'D', '^', 'x']
    handles = []

    fig, ax = pl.subplots(figsize=(10, 6))
    for ib, (background_scen_label, background_scen) in enumerate(background_scens.items()):
        vx_scen_label = background_scen['vx_scen']
        screen_scen_label = background_scen['screen_scen']
        for it, txvx_scen in enumerate(txvx_scens):
            dalys_noTxV = 0
            dalys_TxV = 0
            cost_noTxV = 0
            cost_TxV = 0
            for location in locations:
                if 'intro 2035' in txvx_scen:
                    NoTxV_econdf_counts = delayed_intro_df[(delayed_intro_df.screen_scen == screen_scen_label) & (delayed_intro_df.vx_scen == vx_scen_label)
                                    & (delayed_intro_df.txvx_scen == 'No TxV') & (delayed_intro_df.location == location)].groupby('year')[
                    ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                    'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

                    NoTxV_econdf_means = delayed_intro_df[(delayed_intro_df.screen_scen == screen_scen_label) & (delayed_intro_df.vx_scen == vx_scen_label)
                                        & (delayed_intro_df.txvx_scen == 'No TxV') & (delayed_intro_df.location == location)].groupby('year')[
                        ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                    if discounting:
                        cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_counts['new_cancers'].values)])
                        cancer_deaths = np.array(
                            [i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_counts['new_cancer_deaths'].values)])
                    else:
                        cancers = NoTxV_econdf_counts['new_cancers'].values
                        cancer_deaths = NoTxV_econdf_counts['new_cancer_deaths'].values
                    avg_age_ca_death = np.mean(NoTxV_econdf_means['av_age_cancer_deaths'])
                    avg_age_ca = np.mean(NoTxV_econdf_means['av_age_cancers'])
                    ca_years = avg_age_ca_death - avg_age_ca
                    yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                    yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                    daly_noTxV = yll + yld
                    dalys_noTxV += daly_noTxV
                    total_cost_noTxV = (NoTxV_econdf_counts['new_tx_vaccinations'].values * cost_dict['txv']) + \
                                    (NoTxV_econdf_counts['new_thermal_ablations'].values * cost_dict['ablation']) + \
                                    (NoTxV_econdf_counts['new_leeps'].values * cost_dict['leep']) + \
                                    (NoTxV_econdf_counts['new_cancer_treatments'].values * cost_dict['cancer'])
                    if discounting:
                        cost_noTxV = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_noTxV)])
                    else:
                        cost_noTxV = np.sum(total_cost_noTxV)
                    cost_noTxV += cost_noTxV
                    TxV_econdf_counts = delayed_intro_df[(delayed_intro_df.screen_scen == screen_scen_label) & (delayed_intro_df.vx_scen == vx_scen_label)
                                        & (delayed_intro_df.txvx_scen == txvx_scen) & (delayed_intro_df.location == location)].groupby(
                        'year')[
                        ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                        'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

                    TxV_econdf_means = delayed_intro_df[(delayed_intro_df.screen_scen == screen_scen_label) & (delayed_intro_df.vx_scen == vx_scen_label)
                                                & (delayed_intro_df.txvx_scen == txvx_scen) & (
                                                            delayed_intro_df.location == location)].groupby(
                        'year')[
                        ['av_age_cancer_deaths', 'av_age_cancers']].mean()    
                else:
                    NoTxV_econdf_counts = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                        & (econ_df.txvx_scen == 'No TxV') & (econ_df.location == location)].groupby('year')[
                        ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                        'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

                    NoTxV_econdf_means = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                        & (econ_df.txvx_scen == 'No TxV') & (econ_df.location == location)].groupby('year')[
                        ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                    if discounting:
                        cancers = np.array([i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_counts['new_cancers'].values)])
                        cancer_deaths = np.array(
                            [i / 1.03 ** t for t, i in enumerate(NoTxV_econdf_counts['new_cancer_deaths'].values)])
                    else:
                        cancers = NoTxV_econdf_counts['new_cancers'].values
                        cancer_deaths = NoTxV_econdf_counts['new_cancer_deaths'].values
                    avg_age_ca_death = np.mean(NoTxV_econdf_means['av_age_cancer_deaths'])
                    avg_age_ca = np.mean(NoTxV_econdf_means['av_age_cancers'])
                    ca_years = avg_age_ca_death - avg_age_ca
                    yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                    yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                    daly_noTxV = yll + yld
                    dalys_noTxV += daly_noTxV
                    total_cost_noTxV = (NoTxV_econdf_counts['new_tx_vaccinations'].values * cost_dict['txv']) + \
                                    (NoTxV_econdf_counts['new_thermal_ablations'].values * cost_dict['ablation']) + \
                                    (NoTxV_econdf_counts['new_leeps'].values * cost_dict['leep']) + \
                                    (NoTxV_econdf_counts['new_cancer_treatments'].values * cost_dict['cancer'])
                    if discounting:
                        cost_noTxV = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_noTxV)])
                    else:
                        cost_noTxV = np.sum(total_cost_noTxV)
                    cost_noTxV += cost_noTxV
                
                    TxV_econdf_counts = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                        & (econ_df.txvx_scen == txvx_scen) & (econ_df.location == location)].groupby(
                        'year')[
                        ['new_tx_vaccinations', 'new_thermal_ablations', 'new_leeps',
                        'new_cancer_treatments', 'new_cancers', 'new_cancer_deaths']].sum()

                    TxV_econdf_means = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                                & (econ_df.txvx_scen == txvx_scen) & (
                                                            econ_df.location == location)].groupby(
                        'year')[
                        ['av_age_cancer_deaths', 'av_age_cancers']].mean()

                if discounting:
                    cancers = np.array([i / 1.03 ** t for t, i in enumerate(TxV_econdf_counts['new_cancers'].values)])
                    cancer_deaths = np.array(
                        [i / 1.03 ** t for t, i in enumerate(TxV_econdf_counts['new_cancer_deaths'].values)])
                else:
                    cancers = TxV_econdf_counts['new_cancers'].values
                    cancer_deaths = TxV_econdf_counts['new_cancer_deaths'].values
                avg_age_ca_death = np.mean(TxV_econdf_means['av_age_cancer_deaths'])
                avg_age_ca = np.mean(TxV_econdf_means['av_age_cancers'])
                ca_years = avg_age_ca_death - avg_age_ca
                yld = np.sum(np.sum([0.54 * .1, 0.049 * .5, 0.451 * .3, 0.288 * .1]) * ca_years * cancers)
                yll = np.sum((standard_le - avg_age_ca_death) * cancer_deaths)
                daly_TxV = yll + yld
                dalys_TxV += daly_TxV

                total_cost_TxV = (TxV_econdf_counts['new_tx_vaccinations'].values * cost_dict['txv']) + \
                                (TxV_econdf_counts['new_thermal_ablations'].values * cost_dict['ablation']) + \
                                (TxV_econdf_counts['new_leeps'].values * cost_dict['leep']) + \
                                (TxV_econdf_counts['new_cancer_treatments'].values * cost_dict['cancer'])
                if discounting:
                    cost_TxV = np.sum([i / 1.03 ** t for t, i in enumerate(total_cost_TxV)])
                else:
                    cost_TxV = np.sum(total_cost_TxV)
                cost_TxV += cost_TxV

            dalys_averted = dalys_noTxV - dalys_TxV
            additional_cost = cost_TxV - cost_noTxV
            cost_daly_averted = additional_cost / dalys_averted

            handle, = ax.plot(dalys_averted/1e6, cost_daly_averted, color=colors[ib], marker=markers[it], linestyle = 'None', markersize=15,)
            handles.append(handle)

    legend1 = ax.legend(handles[0:5], ["90/50", "90/0", "Delayed intro", "50% CP", "No immune memory"], bbox_to_anchor= (1.0, 0.75), title='TxV scenario')
    ax.legend([handles[0], handles[5], handles[10]], background_scens.keys(), loc='upper right', title='Background intervention scale-up')
    pl.gca().add_artist(legend1)

    ax.set_xlabel('DALYs averted (millions), 2030-2060')
    ax.set_ylabel('Incremental costs/DALY averted,\n$USD 2030-2060')

    # ax.set_ylim([0, 120])
    # fig.suptitle(f'TxV CEA for {locations}', fontsize=18)
    fig.tight_layout()
    fig_name = f'{ut.figfolder}/CEA.png'
    fig.savefig(fig_name, dpi=100)

    return


def plot_CEA_tnv(locations=None, background_scen=None, txvx_scens=None, hpv_test_cost_range=None):
    
    ut.set_font(size=14)
    econdfs = sc.autolist()
    dfs = sc.autolist()
    for location in locations:
        econdf = sc.loadobj(f'{ut.resfolder}/{location}_nov25_econ.obj')
        econdfs += econdf
        df = sc.loadobj(f'{ut.resfolder}/{location}_nov25.obj')
        dfs += df
        
    econ_df = pd.concat(econdfs)
    df = pd.concat(dfs)

    cost_dict = dict(
        txv=8,
        leep=41.76,
        ablation=11.76,
        cancer=450
    )
    
    TxV_delivery_scens = ['TnV TxV','Mass TxV']
    marker_sizes = [10, 15, 20, 25]
    
    markers = ['s', 'v']
    handles = []

    colors = sc.gridcolors(len(txvx_scens))
    fig, ax = pl.subplots(figsize=(10, 6))
    vx_scen_label = background_scen['vx_scen']
    screen_scen_label = background_scen['screen_scen']
    for it, txvx_scen in enumerate(txvx_scens):
        for itx, txvx_delivery_scen in enumerate(TxV_delivery_scens):
            txvx_scen_label = f'{txvx_delivery_scen}, {txvx_scen}'
            for i_cost, hpv_test_cost in enumerate(hpv_test_cost_range):
                dalys_noTxV = 0
                dalys_TxV = 0
                cost_noTxV = 0
                cost_TxV = 0
                for location in locations:
                    NoTxV_econdf_counts = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                                & (econ_df.txvx_scen == 'No TxV') & (econ_df.location == location)][
                                ['dalys']].sum()
                    
                    NoTxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                                                & (df.txvx_scen == 'No TxV') & (df.location == location)].groupby('year')[
                                ['new_txvx_doses', 'new_screens', 'new_cin_treatments']].sum()


                    
                    dalys_noTxV += NoTxV_econdf_counts['dalys']
                    cost_noTxV = np.sum(NoTxV['new_txvx_doses'].values * cost_dict['txv']) + \
                                            np.sum(NoTxV['new_screens'].values * hpv_test_cost) + \
                                            np.sum(0.55*NoTxV['new_cin_treatments'].values * cost_dict['leep']) + \
                                            np.sum(0.45*NoTxV['new_cin_treatments'].values * cost_dict['ablation']) 
                    cost_noTxV += cost_noTxV
                        
                    TxV_econdf_counts = econ_df[(econ_df.screen_scen == screen_scen_label) & (econ_df.vx_scen == vx_scen_label)
                                                & (econ_df.txvx_scen == txvx_scen_label) & (econ_df.location == location)][
                                ['dalys']].sum()
                    
                    TxV = df[(df.screen_scen == screen_scen_label) & (df.vx_scen == vx_scen_label)
                                                & (df.txvx_scen == txvx_scen_label) & (df.location == location)].groupby('year')[
                                ['new_txvx_doses', 'new_screens', 'new_cin_treatments']].sum()
                    dalys_TxV += TxV_econdf_counts['dalys']          

                    cost_TxV = np.sum(TxV['new_txvx_doses'].values * cost_dict['txv']) + \
                                            np.sum(TxV['new_screens'].values * hpv_test_cost) + \
                                            np.sum(0.55*TxV['new_cin_treatments'].values * cost_dict['leep']) + \
                                            np.sum(0.45*TxV['new_cin_treatments'].values * cost_dict['ablation']) 
                    cost_TxV += cost_TxV

                dalys_averted = dalys_noTxV - dalys_TxV
                additional_cost = cost_TxV - cost_noTxV
                cost_daly_averted = additional_cost / dalys_averted

                handle, = ax.plot(dalys_averted/1e6, cost_daly_averted, color=colors[it], linestyle = 'None', marker=markers[itx], markersize=marker_sizes[i_cost])
                handles.append(handle)

    legend1 = ax.legend([handles[0], handles[4]], ["Test & Vaccinate", "Mass Vaccination"], bbox_to_anchor= (0.8, 1.0), title='TxV Delivery')
    legend2 = ax.legend([handles[0], handles[1], handles[2], handles[3]], hpv_test_cost_range, bbox_to_anchor= (0.5, 1.0), title='HPV test cost', ncol=2)
    ax.legend([handles[0], handles[8], handles[16]], ['90/50', '70/30', '90/0'], loc='upper right', title='TxV Effectiveness')
    pl.gca().add_artist(legend1)
    pl.gca().add_artist(legend2)

    ax.set_xlabel('DALYs averted (millions), 2030-2060')
    ax.set_ylabel('Incremental costs/DALY averted,\n$USD 2030-2060')
    # sc.setylim(ax, 0, 200)
    fig.tight_layout()
    fig_name = f'{ut.figfolder}/CEA_tnv.png'
    fig.savefig(fig_name, dpi=100)
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
        
    ]

    # plot_CEA(
    #         locations=locations,
    #         background_scens={
        
    #             '90-0-0': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
    #             '90-35-70': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 35% sc cov'},
    #             '90-70-90': {'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'HPV, 70% sc cov, 90% tx cov'},
    #         },
        
    #         txvx_scens=[
    #             'Mass TxV, 90/50, age 30',
    #             'Mass TxV, 90/0, age 30',
    #             'Mass TxV, 90/50, age 30, intro 2035',
    #             'Mass TxV, 90/50, age 30, cross-protection',
    #             'Mass TxV, 90/50, age 30, no durable immunity'
                
    #         ]
    #     )
        

    plot_CEA_tnv(
            locations=locations,
            background_scen={'vx_scen': 'Vx, 90% cov, 9-14', 'screen_scen': 'No screening'},
        
            txvx_scens=[
                '90/50, age 30',
                '70/30, age 30',
                '90/0, age 30',
            ],
            
            hpv_test_cost_range = [5,10,15,20]
            
        )
    
    T.toc('Done')