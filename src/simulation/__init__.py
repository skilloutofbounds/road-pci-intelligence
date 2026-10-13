"""
Simulation module for pavement maintenance optimization and budget allocation.
All costs, treatment gains, and deterioration rates are illustrative assumptions.
"""

from .network import generate_synthetic_network, RoadSegmentNetwork
from .engine import SimulationEngine, run_policy_simulation, run_monte_carlo, run_sensitivity_grid

__all__ = [
    "generate_synthetic_network",
    "RoadSegmentNetwork",
    "SimulationEngine",
    "run_policy_simulation",
    "run_monte_carlo",
    "run_sensitivity_grid",
]
