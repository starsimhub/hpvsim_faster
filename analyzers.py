"""
Define custom analyzers for HPVsim
"""

import numpy as np
import pandas as pd
import sciris as sc
import hpvsim as hpv
import math


class dalys(hpv.Analyzer):
    """
    Analyzer for computing DALYs.
    """

    def __init__(self, start=None, life_expectancy=84, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        self.si = None  # Start index - calculated upon initialization based on sim time vector
        self.df = None  # Results dataframe
        self.disability_weights = sc.objdict(
            weights=[
                0.288,
                0.049,
                0.451,
                0.54,
            ],  # From GBD2017 - see Table A2.1 https://www.thelancet.com/cms/10.1016/S2214-109X(20)30022-X/attachment/0f63cf98-5eb9-48eb-af4f-6abe8fdff544/mmc1.pdf
            time_fraction=[0.05, 0.85, 0.09, 0.01],  # Estimates based on durations
        )
        self.life_expectancy = (
            life_expectancy  # Should typically use country-specific values
        )
        return

    @property
    def av_disutility(self):
        """The average disability weight over duration of cancer"""
        dw = self.disability_weights
        len_dw = len(dw.weights)
        return sum([dw.weights[i] * dw.time_fraction[i] for i in range(len_dw)])

    def initialize(self, sim):
        super().initialize(sim)
        if self.start is None:
            self.start = sim["start"]
        self.si = sc.findfirst(sim.res_yearvec, self.start)
        self.npts = len(sim.res_yearvec[self.si :])
        self.years = sim.res_yearvec[self.si :]
        self.yll = np.zeros(self.npts)
        self.yld = np.zeros(self.npts)
        self.dalys = 0
        return

    def apply(self, sim):

        if sim.yearvec[sim.t] >= self.start:
            ppl = sim.people
            li = np.floor(sim.yearvec[sim.t])
            idx = sc.findfirst(self.years, li)

            # Get new people with cancer and add up all their YLL and YLD now (incidence-based DALYs)
            new_cancers = ppl.date_cancerous == sim.t
            new_cancer_inds = hpv.true(new_cancers)
            if len(new_cancer_inds):
                self.yld[idx] += sum(
                    ppl.scale[new_cancer_inds]
                    * ppl.dur_cancer[new_cancers]
                    * self.av_disutility
                )
                age_death = ppl.age[new_cancer_inds] + ppl.dur_cancer[new_cancers]
                years_left = np.maximum(0, self.life_expectancy - age_death)
                self.yll[idx] += sum(ppl.scale[new_cancer_inds] * years_left)

        return

    def finalize(self, sim):
        self.dalys = np.sum(self.yll + self.yld)
        return



class econ_analyzer(hpv.Analyzer):
    """
    Analyzer for feeding into costing/health economic analysis.

    Produces a dataframe by year storing:

        - Resource use: number of vaccines, screens, lesions treated, cancers treated
        - Cases/deaths: number of new cancer cases and cancer deaths
        - Average age of new cases, average age of deaths, average age of noncancer death
    """

    def __init__(self, start=2020, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start = start
        return

    def initialize(self, sim):
        super().initialize(sim)
        columns = [
            "new_hpv_screens",
            "new_vaccinations",
            "new_thermal_ablations",
            "new_leeps",
            "new_cancer_treatments",
        ]

        self.df = dict()
        for col in columns:
            self.df[col] = 0
        return

    def apply(self, sim):
        return

    def finalize(self, sim):
        for idx in range(sim.res_npts):

            # Pull out characteristics of sim to decide what resources we need
            simvals = sim.meta.vals
            scenario_label = simvals.scen
            self.df["new_vaccinations"] += sim.get_intervention(
                    "Routine vx"
                ).n_products_used.values[idx]
            self.df["new_vaccinations"] += sim.get_intervention(
                    "Catchup vx"
                ).n_products_used.values[idx]
            if scenario_label != 'Status quo':
                self.df["new_hpv_screens"] += sim.get_intervention(
                            "screening"
                        ).n_products_used.values[idx]
                self.df["new_thermal_ablations"] += sim.get_intervention(
                        "ablation"
                    ).n_products_used.values[idx]
                self.df["new_leeps"] += sim.get_intervention(
                        "excision"
                    ).n_products_used.values[idx]
                self.df["new_cancer_treatments"] += sim.get_intervention(
                        "radiation"
                    ).n_products_used.values[idx]
            if 'HPV FASTER' in scenario_label:
                # add in HPV FASTER resources
                self.df["new_vaccinations"] += sim.get_intervention(
                    "HPV FASTER vx"
                ).n_products_used.values[idx]
 

        return

