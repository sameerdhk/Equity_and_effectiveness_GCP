# -----------------------------------------------------------------------------
# gcp_model_fb.py
#
# This script defines the core agent-based model for simulating the
# "Fee-based" (FB) policy, which also serves as the
# baseline scenario. The GCPModelFb class is built on the MESA framework and
# orchestrates the interactions between different agents: Aquifer, Field,
# Well, Finance, and Behavior.
#
# The model initializes all agents based on input dictionaries, advances the
# simulation step-by-step (yearly), and collects data on agent states and
# environmental conditions.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
from copy import deepcopy
import gurobipy as gp
import mesa
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import custom components from the PyCHAMP library structure
from ..components.aquifer import Aquifer
from ..components.behavior import Behavior
from ..components.field import Field
from ..components.finance import Finance
from ..components.optimization import Optimization
from ..components.well import Well
from ..utility.util import BaseSchedulerByTypeFiltered, Indicator, TimeRecorder


# --- 2. Define the Main Model Class ---
class GCPModelFb(mesa.Model):
    """
    A Mesa model for simulating the Fee-Based (FB) policy scenarios.

    This model integrates various agent types (fields, wells, behaviors, aquifers)
    to simulate agricultural decisions and their environmental impacts over time.
    It is designed to be highly configurable to test different policy scenarios.

    Attributes
    ----------
    schedule : BaseSchedulerByTypeFiltered
        The scheduler that activates agents in a specific order during each step.
    datacollector : mesa.DataCollector
        Collects and stores data from agents and the model during the simulation.
    gpenv : gp.Env
        The Gurobi optimization environment used by the agents.
    current_year : int
        The current year of the simulation.
    """

    def __init__(
        self,
        crop_options,
        tech_options,
        area_split,
        aquifers_dict,
        fields_dict,
        wells_dict,
        finances_dict,
        behaviors_dict,
        prec_aw_step,
        prec_aw_rolling_step,
        rolling_precipitaion_average=False,
        init_year=2001,
        end_year=2022,
        components=None,
        optimization_class=Optimization,
        show_step=True,
        seed=None,
        shared_config=None,
        gurobi_dict=None,
        **kwargs,
    ):
        """
        Initializes the GCPModelFb instance.
        """
        # --- 2a. Basic Model and Component Setup ---
        # Set default Gurobi parameters if not provided
        if gurobi_dict is None:
            gurobi_dict = {"LogToConsole": 0, "NonConvex": 2, "Presolve": -1}
        # Set default agent component classes if not provided
        if components is None:
            components = {
                "aquifer": Aquifer,
                "field": Field,
                "well": Well,
                "finance": Finance,
                "behavior": Behavior,
            }
        self.running = True  # Required by MESA to control the simulation loop

        # --- 2b. Time and Simulation Parameters ---
        self.time_recorder = TimeRecorder()
        self.components = components
        self.optimization_class = optimization_class
        self.init_year = init_year
        self.start_year = self.init_year + 1
        self.end_year = end_year
        self.total_steps = self.end_year - self.init_year
        self.current_year = self.init_year
        self.t = 0  # Initial step counter
        self.show_step = show_step
        self.rolling_precipitaion_average = rolling_precipitaion_average
        self.seed = seed
        self.rngen = np.random.default_rng(seed) # Random number generator for reproducibility
        self.accumulated_withdrawal = 0 # This will track pumping within a year

        # --- 2c. Store Model Parameters and Input Data ---
        # Time-series data passed as keyword arguments
        self.prec_aw_step = prec_aw_step
        self.prec_aw_rolling_step = prec_aw_rolling_step
        self.crop_price_step = kwargs.get("crop_price_step")
        self.crop_price_avg_step = kwargs.get("crop_price_avg_step")
        self.crop_variable_cost_step = kwargs.get("crop_variable_cost_step")
        self.crop_variable_cost_avg_step = kwargs.get("crop_variable_cost_avg_step")
        self.crop_fixed_cost_step = kwargs.get("crop_fixed_cost_step")
        self.crop_fixed_cost_avg_step = kwargs.get("crop_fixed_cost_avg_step")
        self.irr_depth_step = kwargs.get("irr_depth_step")
        self.field_type_step = kwargs.get("field_type_step")
        # Dimensions for optimization problems
        self.area_split = area_split
        self.crop_options = crop_options
        self.tech_options = tech_options

        # --- 2d. Initialize Scheduler and Gurobi Environment ---
        self.schedule = BaseSchedulerByTypeFiltered(self)
        self.gpenv = gp.Env(empty=True)
        for k, v in gurobi_dict.items():
            self.gpenv.setParam(k, v)
        self.gpenv.start()

        # --- 2e. Apply Shared Configuration Overrides ---
        # Deepcopy ensures original input dicts are not modified during the run
        aquifers_dict, fields_dict, wells_dict, finances_dict, behaviors_dict = (
            deepcopy(aquifers_dict), deepcopy(fields_dict), deepcopy(wells_dict),
            deepcopy(finances_dict), deepcopy(behaviors_dict),
        )
        # Apply any shared configurations to all relevant agents
        if shared_config is not None:
            config_aquifer = shared_config.get("aquifer", {})
            config_field = shared_config.get("field", {})
            config_well = shared_config.get("well", {})
            config_finance = shared_config.get("finance", {})
            config_behavior = shared_config.get("behavior", {})
            for k, v in config_aquifer.items():
                for d in aquifers_dict:
                    aquifers_dict[d][k] = v
            for k, v in config_field.items():
                for d in fields_dict:
                    fields_dict[d][k] = v
            for k, v in config_well.items():
                for d in wells_dict:
                    wells_dict[d][k] = v
            for k, v in config_finance.items():
                for d in finances_dict:
                    finances_dict[d][k] = v
            for k, v in config_behavior.items():
                for d in behaviors_dict:
                    behaviors_dict[d][k] = v

        # --- 2f. Initialize Agents ---
        # Initialize Aquifers (environmental agents)
        self.aquifers = {}
        for aqid, aquifer_dict in aquifers_dict.items():
            agt_aquifer = components["aquifer"](unique_id=aqid, model=self, settings=aquifer_dict)
            self.aquifers[aqid] = agt_aquifer
            self.schedule.add(agt_aquifer)

        # Initialize Fields
        self.fields = {}
        for fid, field_dict in fields_dict.items():
            # If initial crop is a list, randomly choose one
            if isinstance(field_dict["init"]["crop"], list):
                field_dict["init"]["crop"] = self.rngen.choice(field_dict["init"]["crop"])
            # Create a field agent instance
            agt_field = components["field"](
                unique_id=fid, model=self, settings=field_dict,
                irr_freq=field_dict.get("irr_freq"),
                truncated_normal_pars=field_dict.get("truncated_normal_pars"),
                lat=field_dict.get("lat"), lon=field_dict.get("lon"),
                x=field_dict.get("x"), y=field_dict.get("y"), regen=self.rngen,
                field_type_rn=field_dict.get("field_type_rn"),
            )
            self.fields[fid] = agt_field
            self.schedule.add(agt_field)

        # Initialize Wells
        self.wells = {}
        for wid, well_dict in wells_dict.items():
            agt_well = components["well"](unique_id=wid, model=self, settings=well_dict)
            self.wells[wid] = agt_well
            self.schedule.add(agt_well)

        # Initialize Behavior and Finance Agents
        self.max_num_fields_per_agt = 0
        self.max_num_wells_per_agt = 0
        self.behaviors = {}
        self.finances = {}
        for behavior_id, behavior_dict in tqdm(behaviors_dict.items(), desc="Initialize agents"):
            # Each behavior (farmer) agent has a corresponding finance agent
            finance_id = behavior_dict["finance_id"]
            finance_dict = finances_dict[finance_id]
            agt_finance = components["finance"](
                unique_id=f"{finance_id}_{behavior_id}", model=self,
                settings=finance_dict,
            )
            agt_finance.finance_id = finance_id
            self.finances[behavior_id] = agt_finance # Assumes one finance object per behavior agent
            self.schedule.add(agt_finance)

            # Create the behavior agent and link it to its components (fields, wells, finance)
            agt_behavior = components["behavior"](
                unique_id=behavior_id, model=self, settings=behavior_dict,
                fields={fid: self.fields[fid] for i, fid in enumerate(behavior_dict["field_ids"])},
                wells={wid: self.wells[wid] for i, wid in enumerate(behavior_dict["well_ids"])},
                finance=agt_finance, aquifers=self.aquifers,
                optimization_class=self.optimization_class, rngen=self.rngen,
            )
            self.behaviors[behavior_id] = agt_behavior
            self.schedule.add(agt_behavior)
            # Track max number of fields/wells per agent for data collection purposes
            self.max_num_fields_per_agt = max(self.max_num_fields_per_agt, len(behavior_dict["field_ids"]))
            self.max_num_wells_per_agt = max(self.max_num_wells_per_agt, len(behavior_dict["well_ids"]))

        # --- 2g. Set Initial Financial Conditions ---
        # Update financial states based on the initial year's data
        if self.crop_variable_cost_step is not None:
            for _, finance in self.finances.items():
                crop_vcosts = self.crop_variable_cost_step.get(finance.finance_id)
                if crop_vcosts is not None:
                    finance.crop_variable_cost = crop_vcosts[self.current_year]
        if self.crop_fixed_cost_step is not None:
            for _, finance in self.finances.items():
                crop_fcosts = self.crop_fixed_cost_step.get(finance.finance_id)
                if crop_fcosts is not None:
                    finance.crop_fixed_cost = crop_fcosts[self.current_year]
        if self.crop_price_step is not None:
            for _, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                if crop_prices is not None:
                    finance.crop_price = crop_prices[self.current_year]
        if self.crop_price_avg_step is not None:
            for _unique_id, finance in self.finances.items():
                crop_prices_avg = self.crop_price_avg_step.get(finance.finance_id)
                if crop_prices_avg is not None:
                    finance.crop_price_avg = crop_prices_avg[self.current_year]
                    finance.finance_dict['crop_price_avg'] = finance.crop_price_avg
        if self.crop_variable_cost_avg_step is not None:
            for _unique_id, finance in self.finances.items():
                crop_vcosts_avg = self.crop_variable_cost_avg_step.get(finance.finance_id)
                if crop_vcosts_avg is not None:
                    finance.crop_vcost_avg = crop_vcosts_avg[self.current_year]
                    finance.finance_dict['crop_variable_cost_avg'] = finance.crop_vcost_avg
        if self.crop_fixed_cost_avg_step is not None:
            for _unique_id, finance in self.finances.items():
                crop_fcosts_avg = self.crop_fixed_cost_avg_step.get(finance.finance_id)
                if crop_fcosts_avg is not None:
                    finance.crop_fcost_avg = crop_fcosts_avg[self.current_year]
                    finance.finance_dict['crop_fixed_cost_avg'] = finance.crop_fcost_avg

        # --- 2h. Setup Data Collector ---
        def get_agt_attr(attr_str):
            """Helper function for MESA datacollector to safely get nested attributes."""
            def get_nested_attr(obj):
                attrs = attr_str.split(".")
                for attr in attrs:
                    obj = getattr(obj, attr, None)
                    if obj is None:
                        return None
                return obj
            return get_nested_attr
        
        # Define which attributes to collect from each agent at each step
        agent_reporters = {
            "agt_type": get_agt_attr("agt_type"),
            # Field Reporters
            "field_type": get_agt_attr("field_type"), "crop": get_agt_attr("crops"),
            "tech": get_agt_attr("te"), "w": get_agt_attr("w"),
            "irr_vol_per_field": get_agt_attr("irr_vol_per_field"),
            "yield_rate_per_field": get_agt_attr("yield_rate_per_field"),
            "field_area": get_agt_attr("field_area"), "field_type_rn": get_agt_attr("field_type_rn"),
            # Behavior Reporters
            "yield_rate": get_agt_attr("yield_rate"), "profit": get_agt_attr("profit"),
            "profit_per_field": get_agt_attr("avg_profit_per_field"), "pumping_fee": get_agt_attr("finance.cost_p"),
            "revenue": get_agt_attr("finance.rev"), "energy_cost": get_agt_attr("finance.cost_e"),
            "tech_cost": get_agt_attr("finance.tech_cost"), "irr_vol": get_agt_attr("irr_vol"),
            "gp_status": get_agt_attr("gp_status"), "gp_MIPGap": get_agt_attr("gp_MIPGap"),
            "num_fields": get_agt_attr("num_fields"), "num_wells": get_agt_attr("num_wells"),
            "total_field_area": get_agt_attr("total_field_area"),
            # Internal Check
            # "perceived_precipitation": get_agt_attr("dm_sols_1_perceived_precipitation"),
            # "dm_sols_y_y": get_agt_attr("dm_sols_y_y"), "dm_sols_y": get_agt_attr("dm_sols_y"),
            # "dm_sols_profit": get_agt_attr("dm_sols_profit"),
            # "dm_sols_1_profit": get_agt_attr("dm_sols_1_profit"),
            # "dm_sols_irr_depth": get_agt_attr("dm_sols_irr_depth"),
            # "dm_sols_irr_dept_1": get_agt_attr("dm_sols_1_irr_depth"),
            # "dm_sols_crop": get_agt_attr("dm_sols_crop"),
            # "dm_sols_fixed_production_cost": get_agt_attr("dm_sols_fixed_production_cost"),
            # "dm_sols_variable_production_cost": get_agt_attr("dm_sols_variable_production_cost"),
            # "dm_sols_rev": get_agt_attr("dm_sols_rev"),
            # "pre_dm_sols_count": get_agt_attr("pre_dm_sols_count"),
            # Well Reporters
            "water_depth": get_agt_attr("l_wt"), "pumping rate": get_agt_attr("pumping_rate"),
            "energy": get_agt_attr("e"),
            # Aquifer Reporters
            "withdrawal": get_agt_attr("withdrawal"), "GW_st": get_agt_attr("st"),
            "GW_dwl": get_agt_attr("dwl"),
        }
        model_reporters = {}
        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters, agent_reporters=agent_reporters
        )

        # --- 2i. Print Initialization Summary ---
        estimated_sim_dur = self.time_recorder.sec2str(
            self.time_recorder.get_elapsed_time(strf=False) * self.total_steps
        )
        msg = f"""
        Initial year: \t{self.init_year}
        Simulation period:\t{self.start_year} to {self.end_year}
        Number of agents:\t{len(behaviors_dict)}
        Number of aquifers:\t{len(aquifers_dict)}
        Initialization duration:\t{self.time_recorder.get_elapsed_time()}
        Estimated sim duration:\t{estimated_sim_dur}
        """
        print(msg)

    def step(self):
        """Advances the model by one time step (one year)."""
        self.current_year += 1
        self.t += 1
        #Reset the accumulated withdrawal at the start of each year
        self.accumulated_withdrawal = 0
        
        # --- 3a. Update Time-Dependent Data for All Agents ---
        # Update economic conditions for the current year
        if self.crop_variable_cost_step is not None:
            for _, finance in self.finances.items():
                crop_vcosts = self.crop_variable_cost_step.get(finance.finance_id)
                finance.crop_variable_cost = crop_vcosts[self.current_year]
        if self.crop_fixed_cost_step is not None:
            for _, finance in self.finances.items():
                crop_fcosts = self.crop_fixed_cost_step.get(finance.finance_id)
                if crop_fcosts is not None:
                    finance.crop_fixed_cost = crop_fcosts[self.current_year]
        if self.crop_price_step is not None:
            for _, finance in self.finances.items():
                crop_prices = self.crop_price_step.get(finance.finance_id)
                if crop_prices is not None:
                    finance.crop_price = crop_prices[self.current_year]
        if self.crop_price_avg_step is not None:
            crop_prices_avg = self.crop_price_avg_step.get('finance')
            if crop_prices_avg is not None:
                for _unique_id, finance in self.finances.items():
                    year_price_avg = crop_prices_avg.get(self.current_year)
                    if year_price_avg:
                        finance.crop_price_avg = year_price_avg
                        finance.finance_dict['crop_price_avg'] = year_price_avg
        if self.crop_variable_cost_avg_step is not None:
            crop_vcosts_avg = self.crop_variable_cost_avg_step.get('finance')
            if crop_vcosts_avg is not None:
                for _unique_id, finance in self.finances.items():
                    year_vcost_avg = crop_vcosts_avg.get(self.current_year)
                    if year_vcost_avg:
                        finance.crop_vcost_avg = year_vcost_avg
                        finance.finance_dict['crop_variable_cost_avg'] = year_vcost_avg
        if self.crop_fixed_cost_avg_step is not None:
            crop_fcosts_avg = self.crop_fixed_cost_avg_step.get('finance')
            if crop_fcosts_avg is not None:
                for _unique_id, finance in self.finances.items():
                    year_fcost_avg = crop_fcosts_avg.get(self.current_year)
                    if year_fcost_avg:
                        finance.crop_fcost_avg = year_fcost_avg
                        finance.finance_dict['crop_fixed_cost_avg'] = year_fcost_avg

        # --- 3b. Pre-Step Agent Logic ---
        for _behavior_id, behavior in self.behaviors.items():
            # Store decisions from the previous step for analysis
            behavior.pre_dm_sols = behavior.dm_sols
            # Determine field type (e.g., rainfed or irrigated) for the current step
            for fid_, field in behavior.fields.items():
                rn_irr = False
                if rn_irr: # This block is currently inactive but retained for potential future use
                    irr_freq = field.irr_freq
                    rn = self.rngen.uniform(0, 1)
                    if rn <= irr_freq:
                        field.field_type = "optimize"
                        field.field_type_rn = "optimize"
                    else:
                        field.field_type = "rainfed"
                        field.field_type_rn = "rainfed"
                else:
                    field.field_type = field.field_type
                    field.field_type_rn = field.field_type_rn

        # --- 3c. Activate Behavior Agents ---
        # This is the core decision-making step for farmer agents.
        self.schedule.step(agt_type="Behavior")

        # --- 3d. Update Environmental Agents ---
        for aq_id, aquifer in self.aquifers.items():
            # Aggregate total withdrawal from all wells connected to this aquifer
            withdrawal = sum(
                well.withdrawal for _, well in self.wells.items()
                if well.aquifer_id == aq_id
            )
            # Update the aquifer state based on total withdrawal
            aquifer.step(withdrawal)

        # --- 3e. Data Collection and Logging ---
        self.datacollector.collect(self)
        if self.show_step:
            print(f"Year {self.current_year} [{self.t}/{self.total_steps}]\t{self.time_recorder.get_elapsed_time()}\n")

        # Check for simulation termination
        if self.current_year == self.end_year:
            self.running = False
            print("Done!", f"\t{self.time_recorder.get_elapsed_time()}")

    def end(self):
        """Cleans up the Gurobi environment after the simulation finishes."""
        self.gpenv.dispose()

    @staticmethod
    def get_dfs(model):
        """
        Extracts and processes agent data into structured pandas DataFrames.
        """
        df = model.datacollector.get_agent_vars_dataframe().reset_index()
        df["year"] = df["Step"] + model.init_year
        df.set_index("year", inplace=True)

        # Process Field data
        df_fields = df[df["agt_type"] == "Field"].dropna(axis=1, how="all").copy()
        df_fields["field_type"] = np.nan
        df_fields.loc[df_fields["irr_vol_per_field"] == 0, "field_type"] = "rainfed"
        df_fields.loc[df_fields["irr_vol_per_field"] > 0, "field_type"] = "irrigated"
        df_fields["crop"] = [c[0] for c in df_fields["crop"]] # Assumes area_split = 1
        df_fields["irr_depth_per_field"] = (df_fields["irr_vol_per_field"] / df_fields["field_area"]) * 100 # cm

        # Process Well data
        df_wells = df[df["agt_type"] == "Well"].dropna(axis=1, how="all")

        # Process Aquifer data
        df_aquifers = df[df["agt_type"] == "Aquifer"].dropna(axis=1, how="all")

        # Process Behavior data
        df_behaviors = df[df["agt_type"] == "Behavior"].dropna(axis=1, how="all").copy()
        df_behaviors["irr_depth"] = (df_behaviors["irr_vol"] / df_behaviors["total_field_area"]) * 100 # cm

        return df_behaviors, df_fields, df_wells, df_aquifers

    @staticmethod
    def get_df_sys(model, df_behaviors, df_fields, df_wells, df_aquifers):
        """
        Aggregates agent-level data to create a system-level summary DataFrame.
        """
        df_sys = pd.DataFrame()
        # Aggregate aquifer data
        df_sys["GW_st"] = df_aquifers["GW_st"]
        df_sys["withdrawal"] = df_aquifers["withdrawal"]

        # Calculate ratio of rainfed vs. irrigated area
        dff_field_type = df_fields.groupby([df_fields.index, "field_type"])["field_area"].sum()
        total_area = df_fields.groupby(df_fields.index)["field_area"].sum()
        rainfed_area = dff_field_type.unstack().get("rainfed", 0)
        df_sys["rainfed"] = (rainfed_area / total_area).fillna(0)

        # Calculate rainfed ratio based on random number assignment
        dff_field_type_rn = df_fields.groupby([df_fields.index, "field_type_rn"])["field_area"].sum()
        rainfed_area_rn = dff_field_type_rn.unstack().get("rainfed", 0)
        df_sys["rainfed_rn"] = (rainfed_area_rn / total_area).fillna(0)

        # Calculate crop type ratios
        dff_crop = df_fields.groupby([df_fields.index, "crop"])["field_area"].sum().unstack().fillna(0)
        for crop in model.crop_options:
            if crop in dff_crop.columns:
                df_sys[crop] = dff_crop[crop] / total_area
            else:
                df_sys[crop] = 0

        return df_sys

    @staticmethod
    def get_metrices(df_sys, data, targets=None, indicators_list=None):
        """
        Calculates performance metrics by comparing simulation results to reference data.
        """
        if targets is None:
            targets = ["GW_st", "withdrawal", "rainfed", "corn", "sorghum", "soybeans", "wheat", "fallow"]
        if indicators_list is None:
            indicators_list = ["r", "rmse", "kge"]

        indicators = Indicator()
        metrices = []
        for tar in targets:
            metrices.append(
                indicators.cal_indicator_df(
                    x_obv=data[tar], y_sim=df_sys[tar], index_name=tar,
                    indicators_list=indicators_list,
                )
            )
        return pd.concat(metrices)

