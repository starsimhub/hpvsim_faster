'''
Define custom intervention
'''

import hpvsim as hpv
from hpvsim import utils as hpu
from hpvsim import defaults as hpd
from hpvsim import immunity as hpimm
import numpy as np


class TxSegmented(hpv.tx):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.results = {
            'averted_cancers': 0,
            'treatments': 0,
        }

    def administer(self, sim, inds, return_format='dict'):
        '''
        Loop over treatment states to determine those who are successfully treated and clear infection
        '''        

        tx_successful = []  # Initialize list of successfully treated individuals
        people = sim.people

        averted_cancer_inds = []
        self.results['treatments'] += people.scale[inds].sum()  # Count total treatments administered
        
        for state in self.states:  # Loop over states
            for g, genotype in sim['genotype_map'].items():  # Loop over genotypes in the sim

                theseinds = inds[hpu.true(people[state][g, inds])]  # Extract people for whom this state is true for this genotype

                if len(theseinds):
                    df_filter = (self.df.state == state)  # Filter by state
                    if self.ng > 1: df_filter = df_filter & (self.df.genotype == genotype)
                    thisdf = self.df[df_filter]  # apply filter to get the results for this state & genotype

                    # Determine whether treatment is successful
                    efficacy = thisdf.efficacy.values[0]
                    eff_probs = np.full(len(theseinds), efficacy, dtype=hpd.default_float)  # Assign probabilities of treatment success
                    to_eff_treat = hpu.binomial_arr(eff_probs)  # Determine who will have effective treatment
                    eff_treat_inds = theseinds[to_eff_treat]
                    if len(eff_treat_inds):
                        tx_successful += list(eff_treat_inds)
                        people[state][g, eff_treat_inds] = False  # People who get treated have their CINs removed
                        people['cin'][g, eff_treat_inds] = False  # People who get treated have their CINs removed
                        people[f'date_{state}'][g, eff_treat_inds] = np.nan
                        averted_cancers = eff_treat_inds[hpu.defined(people['date_cancerous'][g, eff_treat_inds])]
                        averted_cancer_inds += list(averted_cancers)
                        people[f'date_cancerous'][g, eff_treat_inds] = np.nan

                        # alternatively, clear now:
                        people.susceptible[g, eff_treat_inds] = True
                        people.infectious[g, eff_treat_inds] = False
                        people.inactive[g, eff_treat_inds] = False  # should already be false
                        hpimm.update_peak_immunity(people, eff_treat_inds, imm_pars=people.pars, imm_source=g)  # update immunity
                        people.date_reactivated[g, eff_treat_inds] = np.nan

        averted_cancer_inds = np.array(list(set(averted_cancer_inds)))
        if len(averted_cancer_inds) > 0:
            print('averted_cancer_inds', averted_cancer_inds)
        #     self.results['averted_cancers'] += people.scale[averted_cancer_inds].sum()
        # print('averted_cancer_inds', averted_cancer_inds)
        # self.results['averted_cancers'] += people.scale[averted_cancer_inds].sum()
        tx_successful = np.array(list(set(tx_successful)))
        tx_unsuccessful = np.setdiff1d(inds, tx_successful)
        if return_format == 'dict':
            output = {'successful': tx_successful, 'unsuccessful': tx_unsuccessful}
        elif return_format == 'array':
            output = tx_successful


        return output
