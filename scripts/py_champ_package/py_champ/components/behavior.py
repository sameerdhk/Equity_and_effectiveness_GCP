# -----------------------------------------------------------------------------
# behavior.py
#
# This script defines the Behavior_pure_optimization class, which represents
# a farmer agent in the model. This agent's primary function is to make
# profit-maximizing decisions about cropping and irrigation each year.
#
# This version of the behavior agent is simplified to focus solely on
# economic optimization. It uses a two-step optimization process to first
# determine the optimal crop choice and then the optimal irrigation depth.
#
# Original code by Chung-Yi Lin at Virginia Tech.
# Modified for this project in September 2025.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import mesa
import numpy as np
import gurobipy as gp
from copy import deepcopy


# --- 2. Define the Behavior Agent Class ---
class Behavior(mesa.Agent):
    """
    Simulates a farmer's decision-making process based on pure profit optimization.
    """

    def __init__(
        self,
        unique_id,
        model,
        settings: dict,
        fields: dict,
        wells: dict,
        finance,
        aquifers: dict,
        optimization_class: object,
        **kwargs,
    ):
        """
        Initializes a Behavior agent.
        """
        # --- 2a. MESA and Core Attribute Setup ---
        # Initialize the agent within the MESA framework
        super().__init__(unique_id, model)
        self.agt_type = "Behavior"
        self.optimization_class = optimization_class

        # Load any other keyword arguments passed during initialization
        for k, v in kwargs.items():
            setattr(self, k, v)

        # --- 2b. Load Settings and Assign Assets ---
        # Load all agent-specific settings from the settings dictionary
        self.load_settings(settings)

        # A counter for tracking if an optimization solution was infeasible
        self.pre_dm_sols_count = 0

        # Assign the physical and financial assets managed by this agent
        self.aquifers = aquifers
        self.fields = fields
        self.wells = wells
        self.finance = finance
        self.num_fields = len(fields)
        self.num_wells = len(wells)
        self.total_field_area = sum([field.field_area for _, field in self.fields.items()])

        # --- 2c. Initialize State Variables ---
        # These attributes will be updated at each step of the simulation
        self.t = 0
        self.irr_vol = None
        self.profit = None
        self.avg_profit_per_field = None
        self.yield_rate = None
        self.dm_sols_y_y = None
        self.zero_irrigation_reason = "NA" # For all other policies except PR-I
        self.prior_appropriation_status = None # Needed for PR-II

        # --- 2d. Initial Decision-Making Run ---
        # Create an initial decision-making solutions (dm_sols) dictionary
        # based on the initial state of the fields.
        dm_sols = {}
        for fi, field in self.fields.items():
            dm_sols[fi] = {}
            dm_sols[fi]["i_crop"] = field.i_crop
            dm_sols[fi]["pre_i_crop"] = field.pre_i_crop
            dm_sols[fi]["i_te"] = field.te
            dm_sols[fi]["pre_i_te"] = field.pre_te

        # Run the two-step optimization process to determine the initial decisions
        self.dm_sols_1 = self.make_dm1(dm_sols=dm_sols, init=True)
        self.dm_sols = self.make_dm2(dm_sols=self.dm_sols_1, init=True)

        # Run a simulation based on these initial decisions to set the state for the first year
        if hasattr(self.model, 'withdrawal_cutoff'):
            # This is a PR-I scenario with a dynamic, within-year cutoff.
            self.run_simulation_pr1()
        elif hasattr(self.model, 'senior_farmers_number'):
            # This is a PR-II scenario where the agent's role was pre-determined.
            self.run_simulation_pr2()
        else:
            # This is a Baseline, UR, or FB scenario with no priority logic.
            self.run_simulation()

    def load_settings(self, settings: dict):
        """
        Loads settings from a dictionary into agent attributes.
        """
        self.behavior_ids_in_network = settings["behavior_ids_in_network"]
        self.field_ids = settings["field_ids"]
        self.well_ids = settings["well_ids"]
        self.finance_id = settings["finance_id"]
        self.seniority_id = settings.get("seniority_id")
        self.dm_dict = settings["decision_making"]
        self.wr_dict = settings["water_rights"]
        self.gb_dict = settings["gurobi"]

    def step(self):
        """
        Executes a single time step (one year) for the behavior agent.
        """
        self.t += 1
        
        # --- 3a. Pre-Optimization Policy Check (for PR-II) ---
        if hasattr(self.model, 'senior_farmers_number'):
           self.model.seniority_id_count += 1
           if self.model.seniority_id_count > self.model.senior_farmers_number:
               for fi, field in self.fields.items():
                   field.field_type = 'rainfed'
               self.prior_appropriation_status = True
           else:
               self.prior_appropriation_status = False
               
        # --- 3a.1 Cash-for-Blue counterfactual for enrolled farmers ---
        if getattr(self.model, "cash_for_blue_enabled", False) and getattr(self, "cb_enrolled", False):
            self.compute_cash_for_blue_counterfactual()        
        
        # --- 3b. Core Decision-Making ---
        # The agent makes decisions using a two-step optimization process.
        # Step 1: Optimize crop choice based on forecasted conditions.
        self.dm_sols_1 = self.make_dm1(dm_sols=self.pre_dm_sols)
        # Step 2: Optimize irrigation depth based on the chosen crop and actual conditions.
        self.dm_sols = self.make_dm2(dm_sols=self.dm_sols_1)

        # --- 3c. Handle Optimization Results ---
        # Retrieve optimization status from the results dictionary
        dm_sols = self.dm_sols
        self.gp_status = dm_sols.get("gp_status")
        self.gp_MIPGap = dm_sols.get("gp_MIPGap")
        self.gp_report = dm_sols.get("gp_report")

        # If the optimization was infeasible, the agent reuses the decisions from the previous year.
        if self.gp_report == "Optimal solution is not found.":
            self.dm_sols = self.pre_dm_sols
            self.pre_dm_sols_count = 1
            print("Reverting to previous solutions due to lack of optimal results.")

        # --- 3d. Simulate Outcomes ---
        # Run the simulation based on the final decisions for the year.
        if hasattr(self.model, 'withdrawal_cutoff'):
            # This is a PR-I scenario with a dynamic, within-year cutoff.
            self.run_simulation_pr1()
        elif hasattr(self.model, 'senior_farmers_number'):
            # This is a PR-II scenario where the agent's role was pre-determined.
            self.run_simulation_pr2()
        else:
            # This is a Baseline, UR, or FB scenario with no priority logic.
            self.run_simulation()

        return self

    def _snapshot_agent_state(self, agent):
        """
        Snapshot an agent state without deep-copying the full model object.
        Used only for FB-CB counterfactual rollback.
        """
        state = {}
        for k, v in agent.__dict__.items():
            if k == "model":
                continue
            try:
                state[k] = deepcopy(v)
            except Exception:
                state[k] = v
        return state

    def _restore_agent_state(self, agent, state):
        """
        Restore an agent state after FB-CB counterfactual simulation.
        """
        model_ref = agent.model
        agent.__dict__.clear()
        agent.__dict__.update(state)
        agent.model = model_ref

    def compute_cash_for_blue_counterfactual(self):
        """
        For FB-CB enrolled farmers only:
        estimate realized normal-FB profit and irrigation volume if the farmer
        had not been enrolled.

        The optimizer chooses the counterfactual decision. Then fields, wells,
        and finance are simulated so the counterfactual profit uses current-year
        realized prices/costs, pumping energy, and pumping fees.
        """
        if not getattr(self.model, "cash_for_blue_enabled", False):
            return

        if not getattr(self, "cb_enrolled", False):
            return

        current_year = self.model.current_year
        prec_aw_step = self.model.prec_aw_step
        aquifers = self.aquifers
        fields = self.fields
        wells = self.wells

        original_field_types = {
            fid: field.field_type for fid, field in fields.items()
        }
        field_states = {fid: self._snapshot_agent_state(field) for fid, field in fields.items()}
        well_states = {wid: self._snapshot_agent_state(well) for wid, well in wells.items()}
        finance_state = self._snapshot_agent_state(self.finance)

        try:
            # Temporarily restore normal FB eligibility.
            for _, field in fields.items():
                field.field_type = field.field_type_rn

            # Counterfactual normal-FB decision.
            cf_dm1 = self.make_dm1(dm_sols=self.pre_dm_sols)
            cf_dm2 = self.make_dm2(dm_sols=cf_dm1)

            # ---- Counterfactual field simulation ----
            for fi, field in fields.items():
                irr_depth = cf_dm2[fi]["irr_depth"][:, :, [0]]
                i_crop = cf_dm2[fi]["i_crop"].copy()
                i_te = cf_dm2[fi]["i_te"]

                field.step(
                    irr_depth=irr_depth,
                    i_crop=i_crop,
                    i_te=i_te,
                    prec_aw=prec_aw_step[field.prec_aw_id][current_year],
                )

            # ---- Counterfactual well simulation ----
            allo_r = cf_dm2["allo_r"]
            allo_r_w = cf_dm2["allo_r_w"]
            field_ids = cf_dm2["field_ids"]
            well_ids = cf_dm2["well_ids"]

            cf_irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])

            for k, wid in enumerate(well_ids):
                well = wells[wid]
                withdrawal = cf_irr_vol * allo_r_w[k, 0]
                pumping_rate = sum(
                    [
                        fields[fid].pumping_rate * allo_r[f, k, 0]
                        for f, fid in enumerate(field_ids)
                    ]
                )
                l_pr = sum(
                    [
                        fields[fid].l_pr * allo_r[f, k, 0]
                        for f, fid in enumerate(field_ids)
                    ]
                )
                dwl = aquifers[well.aquifer_id].dwl

                well.step(
                    withdrawal=withdrawal,
                    dwl=dwl,
                    pumping_rate=pumping_rate,
                    l_pr=l_pr,
                )

            # ---- Dry-well adjustment, same logic as actual simulation ----
            at_least_one_dry_well = any(wells[wid].st <= 0.5 for wid in well_ids)

            if at_least_one_dry_well:
                eligible_well_ids = [wid for wid in well_ids if wells[wid].st > 0.5]

                if not eligible_well_ids:
                    # Restore field and well states before re-simulating zero-irrigation case.
                    for fid, field in fields.items():
                        self._restore_agent_state(field, field_states[fid])
                        field.field_type = field.field_type_rn

                    for wid, well in wells.items():
                        self._restore_agent_state(well, well_states[wid])

                    # Re-simulate fields with zero irrigation.
                    for fi, field in fields.items():
                        zero_irr = np.zeros_like(cf_dm2[fi]["irr_depth"][:, :, [0]])
                        field.step(
                            irr_depth=zero_irr,
                            i_crop=cf_dm2[fi]["i_crop"].copy(),
                            i_te=cf_dm2[fi]["i_te"],
                            prec_aw=prec_aw_step[field.prec_aw_id][current_year],
                        )

                    cf_irr_vol = 0.0

                    # Set all wells to zero withdrawal/energy for finance calculation.
                    for _, well in wells.items():
                        well.withdrawal = 0.0
                        well.pumping_rate = 0.0
                        well.e = 0.0

                else:
                    # Restore wells before re-running eligible-well allocation.
                    for wid, well in wells.items():
                        self._restore_agent_state(well, well_states[wid])

                    total_allo_r_w = sum(
                        allo_r_w[k, 0]
                        for k, wid in enumerate(well_ids)
                        if wid in eligible_well_ids
                    )

                    if total_allo_r_w > 0:
                        normalized_allo_r_w = {
                            wid: allo_r_w[k, 0] / total_allo_r_w
                            for k, wid in enumerate(well_ids)
                            if wid in eligible_well_ids
                        }
                    else:
                        normalized_allo_r_w = {
                            wid: 1 / len(eligible_well_ids)
                            for wid in eligible_well_ids
                        }

                    for k, wid in enumerate(well_ids):
                        well = wells[wid]

                        if wid not in eligible_well_ids:
                            well.withdrawal = 0.0
                            well.pumping_rate = 0.0
                            well.e = 0.0
                            continue

                        withdrawal = cf_irr_vol * normalized_allo_r_w[wid]
                        pumping_rate = sum(
                            [
                                fields[fid].pumping_rate * allo_r[f, k, 0]
                                for f, fid in enumerate(field_ids)
                            ]
                        )
                        l_pr = sum(
                            [
                                fields[fid].l_pr * allo_r[f, k, 0]
                                for f, fid in enumerate(field_ids)
                            ]
                        )
                        dwl = aquifers[well.aquifer_id].dwl

                        well.step(
                            withdrawal=withdrawal,
                            dwl=dwl,
                            pumping_rate=pumping_rate,
                            l_pr=l_pr,
                        )

            # Calculate realized counterfactual profit using finance class.
            # This includes current-year prices/costs and pumping fee.
            self.finance.step(fields=fields, wells=wells)

            self.cb_counterfactual_profit = self.finance.profit
            self.cb_counterfactual_irr_vol = cf_irr_vol
            self.cb_counterfactual_dm_sols = cf_dm2
            self.cb_counterfactual_pumping_fee = getattr(self.finance, "cost_p", None)

        finally:
            # Restore actual enrolled state.
            for fid, field in fields.items():
                self._restore_agent_state(field, field_states[fid])
                field.field_type = original_field_types[fid]

            for wid, well in wells.items():
                self._restore_agent_state(well, well_states[wid])

            self._restore_agent_state(self.finance, finance_state)
            
    def apply_cash_for_blue_payout(self):
        """
        Adds cash-for-blue payout only when the model is running FB_CB.
        Does nothing for FB, UR, PR-I, PR-II, R+PR, or baseline.
        """
        if not getattr(self.model, "cash_for_blue_enabled", False):
            return
    
        cb_enrolled = getattr(self, "cb_enrolled", False)
        cb_payout = getattr(self, "cb_payout", 0.0)
    
        self.cb_production_profit = self.finance.profit
    
        if cb_enrolled:
            self.profit = self.cb_production_profit + cb_payout
        else:
            self.profit = self.cb_production_profit
    
        self.cb_total_profit = self.profit

    def run_simulation(self):
        """
        Simulates the physical and financial outcomes based on the agent's decisions.
        """
        # Get current state information from the main model
        current_year = self.model.current_year
        prec_aw_step = self.model.prec_aw_step
        aquifers = self.aquifers
        fields = self.fields
        wells = self.wells
        dm_sols = self.dm_sols

        # Store results from the first-step optimization for later analysis
        dm_sols_1 = self.dm_sols_1
        for fi, field in fields.items():
            self.dm_sols_1_y_y = dm_sols_1[fi]["y_y"]
            self.dm_sols_1_crop = dm_sols_1[fi]["i_crop"]
            self.dm_sols_1_perceived_precipitation = dm_sols_1[fi]["perceived_precipitation"]
            self.dm_sols_1_y = dm_sols_1[fi]["y"]
            irr_depth_1 = dm_sols_1[fi]["irr_depth"][:, :, [0]]
            self.dm_sols_1_irr_depth = irr_depth_1
        self.dm_sols_1_profit = dm_sols_1["profit"]

        # --- 4a. Simulate Fields ---
        # Apply the final decisions (dm_sols) to each field agent
        for fi, field in fields.items():
            self.dm_sols_y_y = dm_sols[fi]["y_y"]
            self.dm_sols_crop = dm_sols[fi]["i_crop"]
            self.dm_sols_perceived_precipitation = dm_sols[fi]["perceived_precipitation"]
            self.dm_sols_y = dm_sols[fi]["y"]
            irr_depth = dm_sols[fi]["irr_depth"][:, :, [0]]
            self.dm_sols_irr_depth = irr_depth
            i_crop = dm_sols[fi]["i_crop"].copy()
            i_te = dm_sols[fi]["i_te"]
            field.step(
                irr_depth=irr_depth, i_crop=i_crop, i_te=i_te,
                prec_aw=prec_aw_step[field.prec_aw_id][current_year],
            )
            # Update the accumulated withdrawal immediately after this field is simulated
            self.model.accumulated_withdrawal += field.irr_vol_per_field

        # --- 4b. Simulate Wells ---
        # Allocate the total irrigation volume among the agent's wells based on optimization results
        allo_r = dm_sols["allo_r"]
        allo_r_w = dm_sols["allo_r_w"]
        field_ids = dm_sols["field_ids"]
        well_ids = dm_sols["well_ids"]
        self.irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])

        for k, wid in enumerate(well_ids):
            well = wells[wid]
            withdrawal = self.irr_vol * allo_r_w[k, 0]
            pumping_rate = sum(
                [
                    fields[fid].pumping_rate * allo_r[f, k, 0]
                    for f, fid in enumerate(field_ids)
                ]
            )
            l_pr = sum(
                [fields[fid].l_pr * allo_r[f, k, 0] for f, fid in enumerate(field_ids)]
            )
            dwl = aquifers[well.aquifer_id].dwl
            well.step(
                withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
            )

        # --- 4c. Re-evaluation if any well runs dry ---
        at_least_one_dry_well = any(wells[wid].st <= 0.5 for wid in well_ids)
        if at_least_one_dry_well:
            eligible_well_ids = [wid for wid in well_ids if wells[wid].st > 0.5]
            if not eligible_well_ids:
                # If all wells are dry, re-simulate fields with zero irrigation
                for fi, field in fields.items():
                    field.step(
                        irr_depth=np.zeros_like(dm_sols[fi]["irr_depth"][:, :, [0]]),
                        i_crop=dm_sols[fi]["i_crop"].copy(), i_te=dm_sols[fi]["i_te"],
                        prec_aw=prec_aw_step[field.prec_aw_id][current_year],
                    )
                    # Update the accumulated withdrawal immediately after this field is simulated
                    self.model.accumulated_withdrawal += field.irr_vol_per_field
                self.irr_vol = 0
                # Update all well withdrawals to zero
                for k, wid in enumerate(well_ids):
                    well = wells[wid]
                    withdrawal = 0
                    pumping_rate = 0
                    l_pr = 0
                    dwl = aquifers[well.aquifer_id].dwl
                    well.step(
                        withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
                    )
            else:
                # If some wells are dry, re-allocate the total irrigation volume among the remaining eligible wells
                total_allo_r_w = sum(allo_r_w[k, 0] for k, wid in enumerate(well_ids) if wid in eligible_well_ids)
                if total_allo_r_w > 0:
                    normalized_allo_r_w = {wid: allo_r_w[k, 0] / total_allo_r_w for k, wid in enumerate(well_ids) if wid in eligible_well_ids}
                else:
                    normalized_allo_r_w = {wid: 1 / len(eligible_well_ids) for wid in eligible_well_ids}
                # Re-simulate well withdrawals for the eligible wells with the new allocation
                for k, wid in enumerate(well_ids):
                    if wid not in eligible_well_ids:
                        continue
                    well = wells[wid]
                    withdrawal = self.irr_vol * normalized_allo_r_w[wid]
                    pumping_rate = sum(
                        [
                            fields[fid].pumping_rate * allo_r[f, k, 0]
                            for f, fid in enumerate(field_ids)
                        ]
                    )
                    l_pr = sum(
                        [fields[fid].l_pr * allo_r[f, k, 0] for f, fid in enumerate(field_ids)]
                    )
                    dwl = aquifers[well.aquifer_id].dwl
                    well.step(
                        withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
                    )

        # --- 4d. Calculate Financial Outcomes ---
        self.finance.step(fields=fields, wells=wells)
        self.dm_sols_fixed_production_cost = dm_sols["fixed_production_cost"]
        self.dm_sols_variable_production_cost = dm_sols["variable_production_cost"]
        self.dm_sols_rev = dm_sols["rev"]
        self.dm_sols_profit = dm_sols["profit"]

        # --- 4e. Store Final Agent-Level Results ---
        self.profit = self.finance.profit
        # For FB-CB only, replace reported profit with production profit + payout.
        # For all other policies, this leaves profit unchanged.        
        if getattr(self.model, "cash_for_blue_enabled", False):
            self.apply_cash_for_blue_payout()
        
        self.avg_profit_per_field = self.profit / len(fields)
        
        self.yield_rate = sum(
            [field.yield_rate_per_field for _, field in fields.items()]
        ) / len(fields)

    def run_simulation_pr1(self):
        """
        A specialized simulation method for the Priority-Based Pumping I (PR-I) policy.
        It checks for an annual withdrawal cutoff before applying irrigation and
        updates the model's accumulated withdrawal for the next agent.
        """
        current_year = self.model.current_year
        prec_aw_step = self.model.prec_aw_step
        aquifers = self.aquifers
        fields = self.fields
        wells = self.wells
        dm_sols = self.dm_sols
        
        # Store results from the first-step optimization for later analysis
        dm_sols_1 = self.dm_sols_1
        for fi, field in fields.items():
            self.dm_sols_1_y_y = dm_sols_1[fi]["y_y"]
            self.dm_sols_1_crop = dm_sols_1[fi]["i_crop"]
            self.dm_sols_1_perceived_precipitation = dm_sols_1[fi]["perceived_precipitation"]
            self.dm_sols_1_y = dm_sols_1[fi]["y"]
            irr_depth_1 = dm_sols_1[fi]["irr_depth"][:, :, [0]]
            self.dm_sols_1_irr_depth = irr_depth_1
        self.dm_sols_1_profit = dm_sols_1["profit"]

        # --- 4a. Simulate Fields with Cutoff Logic ---
        for fi, field in fields.items():
            # Store optimization results for data collection
            self.dm_sols_y_y = dm_sols[fi]["y_y"]
            self.dm_sols_crop = dm_sols[fi]["i_crop"]
            self.dm_sols_perceived_precipitation = dm_sols[fi]["perceived_precipitation"]
            self.dm_sols_y = dm_sols[fi]["y"]
            self.dm_sols_irr_depth = dm_sols[fi]["irr_depth"][:, :, [0]]

            # Store the originally planned irrigation to check if the agent intended to irrigate
            self.store_irr_depth = np.sum(dm_sols[fi]["irr_depth"][:, :, [0]])

            planned_irr_depth = dm_sols[fi]["irr_depth"][:, :, [0]]
            planned_vol = np.sum(planned_irr_depth * field.unit_area * 0.01)  # Convert cm-ha to m-ha

            # --- Apply Cutoff Logic ---
            if current_year >= 2002:
                remaining_volume = self.model.withdrawal_cutoff - self.model.accumulated_withdrawal
                if planned_vol > remaining_volume:
                    if remaining_volume > 0:  # Partial cutoff
                        scale_factor = remaining_volume / planned_vol if planned_vol > 0 else 0
                        irr_depth = planned_irr_depth * scale_factor
                        self.zero_irrigation_reason = 'partially cutoff'
                    else:  # Full cutoff
                        irr_depth = np.zeros_like(planned_irr_depth)
                        self.zero_irrigation_reason = 'cutoff' if self.store_irr_depth != 0 else 'optimization'
                else:
                    irr_depth = planned_irr_depth
                    self.zero_irrigation_reason = 'optimization' if self.store_irr_depth == 0 else 'NA'
            else: # Initialization year
                irr_depth = planned_irr_depth
                self.zero_irrigation_reason = 'optimization' if self.store_irr_depth == 0 else 'NA'

            # Override reason for fields designated as rainfed
            if field.field_type_rn == "rainfed":
                irr_depth = np.zeros_like(planned_irr_depth) # Ensure zero irrigation for rainfed
                self.zero_irrigation_reason = 'rainfed'
            
            # Execute the field step with the final calculated irrigation depth
            field.step(
                irr_depth=irr_depth, i_crop=dm_sols[fi]["i_crop"].copy(), i_te=dm_sols[fi]["i_te"],
                prec_aw=prec_aw_step[field.prec_aw_id][current_year]
            )
            # Update the accumulated withdrawal immediately after this field is simulated
            self.model.accumulated_withdrawal += field.irr_vol_per_field

        # --- 4b. Simulate Wells ---
        # Allocate the total irrigation volume among the agent's wells based on optimization results
        allo_r = dm_sols["allo_r"]
        allo_r_w = dm_sols["allo_r_w"]
        field_ids = dm_sols["field_ids"]
        well_ids = dm_sols["well_ids"]
        self.irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])
        
        for k, wid in enumerate(well_ids):
            well = wells[wid]
            withdrawal = self.irr_vol * allo_r_w[k, 0]
            pumping_rate = sum(
                [
                    fields[fid].pumping_rate * allo_r[f, k, 0]
                    for f, fid in enumerate(field_ids)
                ]
            )
            l_pr = sum(
                [fields[fid].l_pr * allo_r[f, k, 0] for f, fid in enumerate(field_ids)]
            )
            dwl = aquifers[well.aquifer_id].dwl
            well.step(
                withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
            )
        
        # --- 4c. Re-evaluation if any well runs dry ---
        at_least_one_dry_well = any(wells[wid].st <= 0.5 for wid in well_ids)
        if at_least_one_dry_well:
            eligible_well_ids = [wid for wid in well_ids if wells[wid].st > 0.5]
            if not eligible_well_ids:
                # If all wells are dry, re-simulate fields with zero irrigation
                for fi, field in fields.items():
                    field.step(
                        irr_depth=np.zeros_like(dm_sols[fi]["irr_depth"][:, :, [0]]),
                        i_crop=dm_sols[fi]["i_crop"].copy(), i_te=dm_sols[fi]["i_te"],
                        prec_aw=prec_aw_step[field.prec_aw_id][current_year],
                    )
                    # Update the accumulated withdrawal immediately after this field is simulated
                    self.model.accumulated_withdrawal += field.irr_vol_per_field
                self.irr_vol = 0

                # Update all well withdrawals to zero
                for k, wid in enumerate(well_ids):
                    well = wells[wid]
                    withdrawal = 0
                    pumping_rate = 0
                    l_pr = 0
                    dwl = aquifers[well.aquifer_id].dwl
                    well.step(
                        withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
                    )
            else:
                # If some wells are dry, re-allocate the total irrigation volume among the remaining eligible wells
                total_allo_r_w = sum(allo_r_w[k, 0] for k, wid in enumerate(well_ids) if wid in eligible_well_ids)
                if total_allo_r_w > 0:
                    normalized_allo_r_w = {wid: allo_r_w[k, 0] / total_allo_r_w for k, wid in enumerate(well_ids) if wid in eligible_well_ids}
                else:
                    normalized_allo_r_w = {wid: 1 / len(eligible_well_ids) for wid in eligible_well_ids}
                # Re-simulate well withdrawals for the eligible wells with the new allocation
                for k, wid in enumerate(well_ids):
                    if wid not in eligible_well_ids:
                        continue
                    well = wells[wid]
                    withdrawal = self.irr_vol * normalized_allo_r_w[wid]
                    pumping_rate = sum(
                        [
                            fields[fid].pumping_rate * allo_r[f, k, 0]
                            for f, fid in enumerate(field_ids)
                        ]
                    )
                    l_pr = sum(
                        [fields[fid].l_pr * allo_r[f, k, 0] for f, fid in enumerate(field_ids)]
                    )
                    dwl = aquifers[well.aquifer_id].dwl
                    well.step(
                        withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
                    )
        
        # --- 4d. Calculate Financial Outcomes ---
        self.finance.step(fields=fields, wells=wells)
        self.dm_sols_fixed_production_cost = dm_sols["fixed_production_cost"]
        self.dm_sols_variable_production_cost = dm_sols["variable_production_cost"]
        self.dm_sols_rev = dm_sols["rev"]
        self.dm_sols_profit = dm_sols["profit"]
        
        # --- 4e. Store Final Agent-Level Results ---
        self.profit = self.finance.profit
        self.avg_profit_per_field = self.profit / len(fields)
        self.yield_rate = sum(
            [field.yield_rate_per_field for _, field in fields.items()]
        ) / len(fields)

    def run_simulation_pr2(self):
        """
        A specialized simulation method for the Priority-Based Pumping II (PR-II) policy.
        The decision to irrigate is made in the 'step' method before optimization.
        This method primarily simulates the outcome and records the reason for the decision.
        """
        current_year = self.model.current_year
        prec_aw_step = self.model.prec_aw_step
        aquifers = self.aquifers
        fields = self.fields
        wells = self.wells
        dm_sols = self.dm_sols
        
        # Store results from the first-step optimization for later analysis
        dm_sols_1 = self.dm_sols_1
        for fi, field in fields.items():
            self.dm_sols_1_y_y = dm_sols_1[fi]["y_y"]
            self.dm_sols_1_crop = dm_sols_1[fi]["i_crop"]
            self.dm_sols_1_perceived_precipitation = dm_sols_1[fi]["perceived_precipitation"]
            self.dm_sols_1_y = dm_sols_1[fi]["y"]
            irr_depth_1 = dm_sols_1[fi]["irr_depth"][:, :, [0]]
            self.dm_sols_1_irr_depth = irr_depth_1
        self.dm_sols_1_profit = dm_sols_1["profit"]

        # --- 4a. Simulate Fields with Cutoff Logic ---
        for fi, field in fields.items():
            # Store optimization results for data collection
            self.dm_sols_y_y = dm_sols[fi]["y_y"]
            self.dm_sols_crop = dm_sols[fi]["i_crop"]
            self.dm_sols_perceived_precipitation = dm_sols[fi]["perceived_precipitation"]
            self.dm_sols_y = dm_sols[fi]["y"]
            self.dm_sols_irr_depth = dm_sols[fi]["irr_depth"][:, :, [0]]

            # Store the originally planned irrigation to check if the agent intended to irrigate
            self.store_irr_depth = np.sum(dm_sols[fi]["irr_depth"][:, :, [0]])
        
            # Determine the reason for zero irrigation
            if self.store_irr_depth == 0 and not self.prior_appropriation_status:
                self.zero_irrigation_reason = 'optimization'
            elif self.prior_appropriation_status and self.model.seniority_id_count <= 254: #254 is the number of optimize field type
                self.zero_irrigation_reason = 'prior_appropriation'
            elif self.store_irr_depth != 0:
                self.zero_irrigation_reason = 'NA'
            else:
                self.zero_irrigation_reason = 'rainfed'

            # Execute the field step with the decisions from the optimization
            field.step(
                irr_depth=dm_sols[fi]["irr_depth"][:, :, [0]],
                i_crop=dm_sols[fi]["i_crop"].copy(),
                i_te=dm_sols[fi]["i_te"],
                prec_aw=prec_aw_step[field.prec_aw_id][current_year],
            )
            # Update the accumulated withdrawal immediately after this field is simulated
            self.model.accumulated_withdrawal += field.irr_vol_per_field

        # --- 4b. Simulate Wells ---
        # Allocate the total irrigation volume among the agent's wells based on optimization results
        allo_r = dm_sols["allo_r"]
        allo_r_w = dm_sols["allo_r_w"]
        field_ids = dm_sols["field_ids"]
        well_ids = dm_sols["well_ids"]
        self.irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])
        
        for k, wid in enumerate(well_ids):
            well = wells[wid]
            withdrawal = self.irr_vol * allo_r_w[k, 0]
            pumping_rate = sum(
                [
                    fields[fid].pumping_rate * allo_r[f, k, 0]
                    for f, fid in enumerate(field_ids)
                ]
            )
            l_pr = sum(
                [fields[fid].l_pr * allo_r[f, k, 0] for f, fid in enumerate(field_ids)]
            )
            dwl = aquifers[well.aquifer_id].dwl
            well.step(
                withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
            )
        
        # --- 4c. Re-evaluation if any well runs dry ---
        at_least_one_dry_well = any(wells[wid].st <= 0.5 for wid in well_ids)
        if at_least_one_dry_well:
            eligible_well_ids = [wid for wid in well_ids if wells[wid].st > 0.5]
            if not eligible_well_ids:
                # If all wells are dry, re-simulate fields with zero irrigation
                for fi, field in fields.items():
                    field.step(
                        irr_depth=np.zeros_like(dm_sols[fi]["irr_depth"][:, :, [0]]),
                        i_crop=dm_sols[fi]["i_crop"].copy(), i_te=dm_sols[fi]["i_te"],
                        prec_aw=prec_aw_step[field.prec_aw_id][current_year],
                    )
                    # Update the accumulated withdrawal immediately after this field is simulated
                    self.model.accumulated_withdrawal += field.irr_vol_per_field
                self.irr_vol = 0

                # Update all well withdrawals to zero
                for k, wid in enumerate(well_ids):
                    well = wells[wid]
                    withdrawal = 0
                    pumping_rate = 0
                    l_pr = 0
                    dwl = aquifers[well.aquifer_id].dwl
                    well.step(
                        withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
                    )
            else:
                # If some wells are dry, re-allocate the total irrigation volume among the remaining eligible wells
                total_allo_r_w = sum(allo_r_w[k, 0] for k, wid in enumerate(well_ids) if wid in eligible_well_ids)
                if total_allo_r_w > 0:
                    normalized_allo_r_w = {wid: allo_r_w[k, 0] / total_allo_r_w for k, wid in enumerate(well_ids) if wid in eligible_well_ids}
                else:
                    normalized_allo_r_w = {wid: 1 / len(eligible_well_ids) for wid in eligible_well_ids}
                # Re-simulate well withdrawals for the eligible wells with the new allocation
                for k, wid in enumerate(well_ids):
                    if wid not in eligible_well_ids:
                        continue
                    well = wells[wid]
                    withdrawal = self.irr_vol * normalized_allo_r_w[wid]
                    pumping_rate = sum(
                        [
                            fields[fid].pumping_rate * allo_r[f, k, 0]
                            for f, fid in enumerate(field_ids)
                        ]
                    )
                    l_pr = sum(
                        [fields[fid].l_pr * allo_r[f, k, 0] for f, fid in enumerate(field_ids)]
                    )
                    dwl = aquifers[well.aquifer_id].dwl
                    well.step(
                        withdrawal=withdrawal, dwl=dwl, pumping_rate=pumping_rate, l_pr=l_pr
                    )
        
        # --- 4d. Calculate Financial Outcomes ---
        self.finance.step(fields=fields, wells=wells)
        self.dm_sols_fixed_production_cost = dm_sols["fixed_production_cost"]
        self.dm_sols_variable_production_cost = dm_sols["variable_production_cost"]
        self.dm_sols_rev = dm_sols["rev"]
        self.dm_sols_profit = dm_sols["profit"]
        
        # --- 4e. Store Final Agent-Level Results ---
        self.profit = self.finance.profit
        self.avg_profit_per_field = self.profit / len(fields)
        self.yield_rate = sum(
            [field.yield_rate_per_field for _, field in fields.items()]
        ) / len(fields)
        
    def make_dm1(self, dm_sols, init=False):
        """
        First step of the decision-making process: optimizing crop choice.
        This step uses forecasted (rolling average) precipitation.
        """
        # --- 5a. Initialize Optimization Model ---
        current_year = self.model.current_year
        aquifers, fields, wells, dm_dict = self.aquifers, self.fields, self.wells, self.dm_dict
        dm = self.optimization_class(
            unique_id=self.unique_id, log_to_console=self.gb_dict.get("LogToConsole"),
            gpenv=self.model.gpenv,
        )
        dm.setup_ini_model(
            target=dm_dict["target"], horizon=dm_dict["horizon"], area_split=self.model.area_split,
            crop_options=self.model.crop_options, tech_options=self.model.tech_options,
            approx_horizon=False, current_year = current_year, gurobi_kwargs={},
        )

        # --- 5b. Set Up Constraints for Each Field, Well, and Water Right ---
        for i, (fi, field) in enumerate(fields.items()):
            dm_sols_fi = dm_sols[fi]

            if self.model.rolling_precipitaion_average:
                prec_aw = self.model.prec_aw_rolling_step[field.prec_aw_id][current_year]
            else:
                prec_aw = self.perceived_prec_aw[fi][current_year]

            if init:
                # During the model's first year, use actual precipitation data for the setup.
                dm.setup_constr_field(
                    field_id=fi, field_area=field.field_area,
                    prec_aw=self.model.prec_aw_step[field.prec_aw_id][current_year],
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"], pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type, i_crop=None, i_rainfed=None, i_te=None,
                )
            else:
                # For all subsequent years, use the forecasted precipitation.
                dm.setup_constr_field(
                    field_id=fi, field_area=field.field_area, prec_aw=prec_aw,
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"], pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type, i_crop=None, i_rainfed=None, i_te=None,
                )

        for wi, well in wells.items():
            proj_dwl = np.mean(aquifers[well.aquifer_id].dwl_list[-dm_dict["n_dwl"] :])
            dm.setup_constr_well(
                well_id=wi, dwl=proj_dwl, st=well.st, l_wt=well.l_wt, r=well.r, k=well.k,
                sy=well.sy, eff_pump=well.eff_pump, eff_well=well.eff_well,
                pumping_days=well.pumping_days, pumping_capacity=well.pumping_capacity,
                rho=well.rho, g=well.g,
            )

        wr_dict = self.wr_dict if init else dm_sols["water_rights"]
        for wr_id, v in self.wr_dict.items():
            if v["status"]:
                wr_args = wr_dict.get(wr_id)
                if wr_args is None:
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr_depth=v["wr_depth"],
                        applied_field_ids=v["applied_field_ids"], time_window=v["time_window"],
                        remaining_tw=v["remaining_tw"], remaining_wr=v["remaining_wr"],
                        tail_method=v["tail_method"],
                    )
                else:
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr_depth=wr_args["wr_depth"],
                        applied_field_ids=wr_args["applied_field_ids"],
                        time_window=wr_args["time_window"], remaining_tw=wr_args["remaining_tw"],
                        remaining_wr=wr_args["remaining_wr"], tail_method=wr_args["tail_method"],
                    )

        # --- 5c. Finalize and Solve the Optimization Problem ---
        dm.setup_constr_finance(self.finance.finance_dict)
        dm.setup_obj(alpha_dict=None)
        dm.finish_setup(display_summary=dm_dict["display_summary"])
        dm.solve(
            keep_gp_model=True, keep_gp_output=True,
            display_report=dm_dict["display_report"], **self.gb_dict,
        )

        dm_sols = dm.sols
        if dm.model.Status == gp.GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif dm.model.Status == gp.GRB.UNBOUNDED:
            print("Model is unbounded.")
        elif dm.model.Status == gp.GRB.INF_OR_UNBD:
            print("Model may be infeasible or unbounded.")

        dm.depose_gp_env()
        return dm_sols

    def make_dm2(self, dm_sols, init=False):
        """
        Second step of decision-making: optimizing irrigation depth with a fixed crop choice.
        This step uses the actual precipitation for the current year.
        """
        # --- 6a. Initialize Optimization Model ---
        current_year = self.model.current_year
        aquifers, fields, wells, dm_dict = self.aquifers, self.fields, self.wells, self.dm_dict
        dm = self.optimization_class(
            unique_id=self.unique_id, log_to_console=self.gb_dict.get("LogToConsole"),
            gpenv=self.model.gpenv,
        )
        dm.setup_ini_model(
            target=dm_dict["target"], horizon=dm_dict["horizon"], area_split=self.model.area_split,
            crop_options=self.model.crop_options, tech_options=self.model.tech_options,
            approx_horizon=False, current_year = current_year, gurobi_kwargs={},
        )

        # --- 6b. Set Up Constraints ---
        for i, (fi, field) in enumerate(fields.items()):
            dm_sols_fi = dm_sols[fi]
            i_crop = dm_sols[fi]["i_crop"].copy()

            if init:
                # During initialization, the crop choice is taken as given from the first step.
                dm.setup_constr_field(
                    field_id=fi, field_area=field.field_area,
                    prec_aw=self.model.prec_aw_step[field.prec_aw_id][current_year],
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"], pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type, i_crop=i_crop, i_rainfed=None, i_te=None,
                )
            else:
                # For regular steps, the crop choice is fixed from the first optimization step (make_dm1).
                dm.setup_constr_field(
                    field_id=fi, field_area=field.field_area,
                    prec_aw=self.model.prec_aw_step[field.prec_aw_id][current_year],
                    water_yield_curves=field.water_yield_curves,
                    tech_pumping_rate_coefs=field.tech_pumping_rate_coefs,
                    pre_i_crop=dm_sols_fi["i_crop"], pre_i_te=dm_sols_fi["i_te"],
                    field_type=field.field_type, i_crop=i_crop, i_rainfed=None, i_te=None,
                )

        for wi, well in wells.items():
            proj_dwl = np.mean(aquifers[well.aquifer_id].dwl_list[-dm_dict["n_dwl"] :])
            dm.setup_constr_well(
                well_id=wi, dwl=proj_dwl, st=well.st, l_wt=well.l_wt, r=well.r, k=well.k,
                sy=well.sy, eff_pump=well.eff_pump, eff_well=well.eff_well,
                pumping_days=well.pumping_days, pumping_capacity=well.pumping_capacity,
                rho=well.rho, g=well.g,
            )

        wr_dict = self.wr_dict if init else dm_sols["water_rights"]
        for wr_id, v in self.wr_dict.items():
            if v["status"]:
                wr_args = wr_dict.get(wr_id)
                if wr_args is None:
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr_depth=v["wr_depth"],
                        applied_field_ids=v["applied_field_ids"], time_window=v["time_window"],
                        remaining_tw=v["remaining_tw"], remaining_wr=v["remaining_wr"],
                        tail_method=v["tail_method"],
                    )
                else:
                    dm.setup_constr_wr(
                        water_right_id=wr_id, wr_depth=wr_args["wr_depth"],
                        applied_field_ids=wr_args["applied_field_ids"],
                        time_window=wr_args["time_window"], remaining_tw=wr_args["remaining_tw"],
                        remaining_wr=wr_args["remaining_wr"], tail_method=wr_args["tail_method"],
                    )

        # --- 6c. Finalize and Solve the Optimization Problem ---
        dm.setup_constr_finance(self.finance.finance_dict)
        dm.setup_obj(alpha_dict=None)
        dm.finish_setup(display_summary=dm_dict["display_summary"])
        dm.solve(
            keep_gp_model=True, keep_gp_output=True,
            display_report=dm_dict["display_report"], **self.gb_dict,
        )

        dm_sols = dm.sols
        if dm.model.Status == gp.GRB.INFEASIBLE:
            print("Model is infeasible.")
        elif dm.model.Status == gp.GRB.UNBOUNDED:
            print("Model is unbounded.")
        elif dm.model.Status == gp.GRB.INF_OR_UNBD:
            print("Model may be infeasible or unbounded.")

        dm.depose_gp_env()
        return dm_sols