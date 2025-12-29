# -----------------------------------------------------------------------------
# finance.py
#
# This script defines the Finance class, which simulates the financial aspects
# for a farmer agent. It calculates revenue from crop sales and subtracts various
# costs, including production, energy for pumping, and operational costs for
# irrigation technology, to determine the final profit.
#
# Original code by Chung-Yi Lin at Virginia Tech.
# Modified for this project in September 2025.
# -----------------------------------------------------------------------------

# --- 1. Import Libraries ---
import mesa
import numpy as np


# --- 2. Define the Finance Agent Class ---
class Finance(mesa.Agent):
    """
    Simulates the financial calculations for a farmer agent for one time step.

    Parameters
    ----------
    unique_id : int
        A unique identifier for this agent.
    model
        The model instance to which this agent belongs.
    settings : dict
        A dictionary containing financial settings, including:
        - 'energy_price' : float [1e4 $/PJ]
        - 'crop_price' : dict[crop -> $/bu]
        - 'crop_fixed_cost' : dict[crop -> 1e4 $ per unit]
        - 'crop_variable_cost' : dict[crop -> 1e4 $ per unit]
        - 'irr_tech_operational_cost' : dict[tech -> 1e4 $]
        - 'irr_tech_change_cost' : dict[(from_tech, to_tech) -> 1e4 $]
        - 'crop_change_cost' : dict[(from_crop, to_crop) -> 1e4 $]

    Attributes
    ----------
    agt_type : str
        The type of the agent, set to 'Finance'.
    profit : float or None
        The profit calculated for the current step [1e4 $].
    y : float or None
        The total yield from all fields [1e4 bu].
    t : int
        The current time step.
    """

    def __init__(self, unique_id, model, settings: dict):
        """Initializes the Finance agent."""
        super().__init__(unique_id, model)
        self.agt_type = "Finance"

        self.load_settings(settings)

        # Initialize attributes that will be calculated during each step
        self.cost_e = None
        self.cost_tech = None
        self.tech_change_cost_total = None
        self.crop_change_cost_total = None
        self.profit = None
        self.y = None
        self.rev = None
        self.t = 0

    def load_settings(self, settings: dict):
        """
        Loads financial settings from a dictionary with robust defaults.
        """
        self.finance_dict = settings

        self.energy_price = settings["energy_price"]
        self.crop_price = settings["crop_price"]
        self.crop_fixed_cost = settings["crop_fixed_cost"]
        self.crop_variable_cost = settings["crop_variable_cost"]

        # Safely load cost maps, defaulting to an empty dictionary if a key is
        # missing or its value is None. This prevents errors during runtime.
        self.pumping_fee = settings.get("pumping_fee", 0) or 0
        self.irr_tech_operational_cost = settings.get("irr_tech_operational_cost", {}) or {}
        self.irr_tech_change_cost = settings.get("irr_tech_change_cost", {}) or {}
        self.crop_change_cost = settings.get("crop_change_cost", {}) or {}

    def step(self, fields: dict, wells: dict) -> float:
        """
        Performs one financial step by aggregating yields and costs to calculate profit.
        - Aggregates total yield and energy consumption.
        - Computes operational, energy, technology change, and crop change costs.
        - Computes total revenue and production costs.
        - Calculates the final net profit.
        """
        self.t += 1

        # Aggregate yield and energy use from all associated fields and wells
        y = sum([field.y for _, field in fields.items()])
        for _, well in wells.items():
            if well.withdrawal == 0:
                well.e = 0
        e = sum([well.e for _, well in wells.items()])
        temp_crop_cost = sum([field.temp_crop_cost for _, field in fields.items()])

        # Operational cost is only incurred if irrigation is used
        cost_tech = sum(
            [
                self.irr_tech_operational_cost.get(field.te, 0)
                if field.irr_vol_per_field > 0
                else 0
                for _, field in fields.items()
            ]
        )

        crop_options = self.model.crop_options

        # Calculate costs associated with changing technology or crops
        tech_change_cost_total = 0
        crop_change_cost_total = 0
        for _, field in fields.items():
            # Calculate technology change cost
            key_te = (field.pre_te, field.te)
            tech_change_cost_total += self.irr_tech_change_cost.get(key_te, 0)

            # Calculate crop change cost by detecting the 'from' and 'to' crops
            i_crop = field.i_crop
            pre_i_crop = field.pre_i_crop
            cc = (i_crop - pre_i_crop)[:, :, 0]

            for s in range(cc.shape[0]):
                ccc = cc[s, :]
                fr = int(np.argmin(ccc))
                to = int(np.argmax(ccc))
                # A change is only registered if one crop is removed (-1) and another is added (+1)
                if ccc[fr] == -1 and ccc[to] == 1:
                    key_crop = (crop_options[fr], crop_options[to])
                    crop_change_cost_total += self.crop_change_cost.get(key_crop, 0)

        # Calculate total energy cost
        cost_e = e * self.energy_price
        
        # This calculation is universal. If pumping_fee is 0, cost_p will be 0.
        irr_vol = sum([field.irr_vol_per_field for _, field in fields.items()])   #m-ha
        if self.model.current_year >= 2002:
            cost_p = self.pumping_fee * irr_vol
        else:
           cost_p = 0 # No fee during the initialization year 

        # Calculate revenue and production costs
        rev = sum(
            [
                y[i, j, :] * self.crop_price[c]
                for i in range(y.shape[0])
                for j, c in enumerate(crop_options)
            ]
        )[0]

        fixed_production_cost = sum(
            [
                temp_crop_cost[i, j, :] * self.crop_fixed_cost[c]
                for i in range(y.shape[0])
                for j, c in enumerate(crop_options)
            ]
        )[0]

        variable_production_cost = sum(
            [
                temp_crop_cost[i, j, :] * self.crop_variable_cost[c]
                for i in range(y.shape[0])
                for j, c in enumerate(crop_options)
            ]
        )[0]

        profit = (
            rev
            - fixed_production_cost
            - variable_production_cost
            - cost_e
            - cost_p
            - cost_tech
            - tech_change_cost_total
            - crop_change_cost_total
        )

        # Store all calculated financial values as agent attributes
        self.y = y
        self.rev = rev
        self.cost_e = cost_e
        self.cost_p = cost_p
        self.cost_tech = cost_tech
        self.tech_change_cost_total = tech_change_cost_total
        self.crop_change_cost_total = crop_change_cost_total
        self.profit = profit

        return profit