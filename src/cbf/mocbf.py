"""
Multi-Output Control Barrier Function Safety Filter for ACC

This module implements the MO-CBF safety layer used to filter the
reinforcement-learning throttle and brake commands.

Safety constraint:
    h(x) = d - T_h * v_ego

where:
    d       : distance to the lead vehicle [m]
    T_h     : desired time headway [s]
    v_ego   : ego vehicle longitudinal speed [m/s]

CBF condition:
    h_dot + k_cbf * h >= 0

This condition is converted into a maximum allowable drive torque.
If the RL throttle/brake action violates the safety condition, the
filter modifies the action minimally to satisfy the CBF constraint.

Author: Saeedeh Lohrasbi
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Optional, Tuple

import numpy as np


logger = logging.getLogger(__name__)


class InterventionType(str, Enum):
    """Types of CBF interventions."""

    NONE = "none"
    THROTTLE_CLAMP = "throttle_clamp"
    BRAKE_ENFORCEMENT = "brake_enforcement"
    REQUIRED_THROTTLE_RELEASE = "required_throttle_release"


@dataclass
class VehicleParameters:
    """Vehicle and CBF parameters."""

    mass: float = 1631.0
    rolling_resistance: float = 0.01
    drag_coefficient: float = 0.25
    frontal_area: float = 2.19
    air_density: float = 1.225
    gravity: float = 9.81
    wheel_radius: float = 0.32
    gear_ratio: float = 1.0
    time_headway: float = 2.0
    cbf_gain: float = 0.5
    brake_gain: float = 50.0

    max_throttle_percent: float = 100.0
    max_brake_level: float = 150.0


@dataclass
class CBFInput:
    """Input state and RL action for the MO-CBF safety filter."""

    throttle_rl: float
    brake_rl: float
    leader_distance: float
    ego_speed: float
    leader_speed: float
    current_rpm: float


@dataclass
class CBFOutput:
    """Output of the MO-CBF safety filter."""

    safe_action: np.ndarray
    safe_throttle: float
    safe_brake: float
    intervention_type: InterventionType
    barrier_value: float
    torque_threshold: float
    required_brake_force: float
    action_difference_l1: float


class EngineMap:
    """
    Engine throttle-torque map.

    The map returns engine torque [Nm] for a given RPM and throttle level.
    Throttle is represented in percent, from 0 to 100.
    """

    def __init__(
        self,
        rpm_values: Optional[np.ndarray] = None,
        throttle_levels: Optional[np.ndarray] = None,
        torque_values: Optional[np.ndarray] = None,
    ) -> None:
        self.rpm_values = (
            rpm_values
            if rpm_values is not None
            else np.array(
                [
                    600,
                    800,
                    1000,
                    1500,
                    2000,
                    2500,
                    3000,
                    3500,
                    4000,
                    4500,
                    5000,
                    5600,
                    6000,
                    6500,
                ],
                dtype=float,
            )
        )

        self.throttle_levels = (
            throttle_levels
            if throttle_levels is not None
            else np.array([0, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
        )

        self.torque_values = (
            torque_values
            if torque_values is not None
            else np.array(
                [
                    [-40, 32, 36, 40, 53, 67, 80, 93, 107, 120],
                    [-43, 40, 45, 50, 67, 83, 100, 117, 133, 150],
                    [-47, 46, 52, 58, 78, 97, 117, 136, 156, 175],
                    [-53, 50, 57, 63, 84, 106, 127, 148, 169, 190],
                    [-57, 54, 60, 67, 89, 111, 133, 156, 178, 200],
                    [-60, 22, 45, 67, 89, 112, 135, 157, 178, 202],
                    [-63, 23, 46, 68, 91, 114, 137, 159, 182, 205],
                    [-70, 23, 47, 70, 93, 117, 140, 163, 187, 210],
                    [-73, 25, 50, 75, 100, 125, 150, 175, 200, 225],
                    [-77, 25, 50, 75, 100, 125, 150, 175, 200, 225],
                    [-80, 24, 48, 72, 96, 119, 143, 167, 191, 215],
                    [-83, 22, 44, 65, 87, 109, 131, 152, 174, 196],
                    [-90, 20, 40, 60, 80, 100, 120, 140, 160, 180],
                    [-100, 18, 36, 53, 71, 89, 107, 124, 142, 160],
                ],
                dtype=float,
            )
        )

    def torque(self, rpm: float, throttle_percent: float) -> float:
        """
        Return engine torque [Nm] for a given RPM and throttle percentage.
        """

        rpm = float(rpm)
        throttle_percent = float(
            np.clip(
                throttle_percent,
                self.throttle_levels.min(),
                self.throttle_levels.max(),
            )
        )

        torque_at_rpm = np.array(
            [
                np.interp(rpm, self.rpm_values, self.torque_values[:, j])
                for j in range(len(self.throttle_levels))
            ],
            dtype=float,
        )

        return float(np.interp(throttle_percent, self.throttle_levels, torque_at_rpm))

    def throttle_from_torque(self, rpm: float, target_torque: float) -> float:
        """
        Approximate throttle percentage that produces the target torque.

        This is used when the CBF computes a maximum allowable torque and
        the RL throttle command must be clamped.
        """

        throttle_grid = np.linspace(0.0, 100.0, 101)
        torque_grid = np.array([self.torque(rpm, t) for t in throttle_grid])

        sorted_indices = np.argsort(torque_grid)
        torque_sorted = torque_grid[sorted_indices]
        throttle_sorted = throttle_grid[sorted_indices]

        target_torque = float(np.clip(target_torque, torque_sorted.min(), torque_sorted.max()))

        return float(np.interp(target_torque, torque_sorted, throttle_sorted))


class MOCBF:
    """
    Multi-Output CBF safety filter for throttle and brake actions.

    The safety filter receives an RL action:

        a_RL = [throttle, brake]

    where both values are normalized in [0, 1].

    It returns:

        a_safe = [safe_throttle, safe_brake]

    also normalized in [0, 1].

    The filter modifies the action only when needed to satisfy the CBF
    safety condition.
    """

    def __init__(
        self,
        vehicle_params: Optional[VehicleParameters] = None,
        engine_map: Optional[EngineMap] = None,
    ) -> None:
        self.params = vehicle_params if vehicle_params is not None else VehicleParameters()
        self.engine_map = engine_map if engine_map is not None else EngineMap()

    def barrier_value(self, leader_distance: float, ego_speed: float) -> float:
        """
        Compute h(x) = d - T_h * v_ego.

        Positive h(x) indicates satisfaction of the nominal time-headway safety set.
        Negative h(x) indicates that the vehicle is outside the defined safe set.
        """

        return float(leader_distance - self.params.time_headway * ego_speed)

    def rolling_force(self) -> float:
        """Return rolling resistance force [N]."""

        p = self.params
        return float(p.mass * p.gravity * p.rolling_resistance)

    def drag_force(self, ego_speed: float) -> float:
        """Return aerodynamic drag force [N]."""

        p = self.params
        return float(0.5 * p.air_density * p.drag_coefficient * p.frontal_area * ego_speed**2)

    def resistance_force(self, ego_speed: float) -> float:
        """Return total resistance force [N]."""

        return self.rolling_force() + self.drag_force(ego_speed)

    def cbf_torque_threshold(
        self,
        leader_distance: float,
        ego_speed: float,
        leader_speed: float,
    ) -> float:
        """
        Compute maximum allowable drive torque [Nm].

        The CBF condition is:

            h_dot + k_cbf h >= 0

        with:

            h = d - T_h v_ego

        This yields a bound on the allowable longitudinal acceleration,
        which is converted into a bound on engine torque.
        """

        p = self.params

        resistance = self.resistance_force(ego_speed)

        net_force_threshold = resistance + (p.mass / p.time_headway) * (
            p.cbf_gain * leader_distance
            + leader_speed
            - (p.cbf_gain * p.time_headway + 1.0) * ego_speed
        )

        torque_threshold = (net_force_threshold * p.wheel_radius) / p.gear_ratio

        return float(torque_threshold)

    def cbf_brake_threshold(
        self,
        leader_distance: float,
        ego_speed: float,
        leader_speed: float,
        current_rpm: float,
    ) -> float:
        """
        Compute the minimum brake force required when throttle is zero.

        If the allowable drive torque is negative, braking may be required.
        This method converts that condition into a minimum brake-force demand.
        """

        p = self.params

        engine_torque_at_zero = self.engine_map.torque(current_rpm, 0.0)
        engine_drag_force = (engine_torque_at_zero * p.gear_ratio) / p.wheel_radius

        drive_torque_threshold = self.cbf_torque_threshold(
            leader_distance=leader_distance,
            ego_speed=ego_speed,
            leader_speed=leader_speed,
        )

        allowable_drive_force = (drive_torque_threshold * p.gear_ratio) / p.wheel_radius
        required_brake_force = engine_drag_force - allowable_drive_force

        return float(required_brake_force)

    def filter_action(self, cbf_input: CBFInput) -> CBFOutput:
        """
        Apply the MO-CBF safety filter.

        Parameters
        ----------
        cbf_input:
            RL action and vehicle state.

        Returns
        -------
        CBFOutput:
            Safe action, intervention type, barrier value, and diagnostics.
        """

        p = self.params

        throttle_rl_percent = float(
            np.clip(
                cbf_input.throttle_rl * p.max_throttle_percent,
                0.0,
                p.max_throttle_percent,
            )
        )

        brake_rl_level = float(
            np.clip(
                cbf_input.brake_rl * p.max_brake_level,
                0.0,
                p.max_brake_level,
            )
        )

        torque_threshold = self.cbf_torque_threshold(
            leader_distance=cbf_input.leader_distance,
            ego_speed=cbf_input.ego_speed,
            leader_speed=cbf_input.leader_speed,
        )

        required_brake_force = self.cbf_brake_threshold(
            leader_distance=cbf_input.leader_distance,
            ego_speed=cbf_input.ego_speed,
            leader_speed=cbf_input.leader_speed,
            current_rpm=cbf_input.current_rpm,
        )

        h_value = self.barrier_value(
            leader_distance=cbf_input.leader_distance,
            ego_speed=cbf_input.ego_speed,
        )

        intervention = InterventionType.NONE

        rl_prefers_throttle = throttle_rl_percent >= brake_rl_level

        if rl_prefers_throttle:
            if torque_threshold < 0.0:
                safe_throttle_percent = 0.0

                if required_brake_force <= 0.0:
                    safe_brake_level = 0.0
                    intervention = InterventionType.REQUIRED_THROTTLE_RELEASE
                else:
                    min_brake_level = required_brake_force / p.brake_gain
                    safe_brake_level = max(min_brake_level, brake_rl_level)
                    intervention = InterventionType.BRAKE_ENFORCEMENT

            else:
                torque_rl = self.engine_map.torque(
                    cbf_input.current_rpm,
                    throttle_rl_percent,
                )

                if torque_rl <= torque_threshold:
                    safe_throttle_percent = throttle_rl_percent
                    safe_brake_level = 0.0
                    intervention = InterventionType.NONE
                else:
                    safe_throttle_percent = self.engine_map.throttle_from_torque(
                        cbf_input.current_rpm,
                        torque_threshold,
                    )
                    safe_brake_level = 0.0
                    intervention = InterventionType.THROTTLE_CLAMP

        else:
            safe_throttle_percent = 0.0

            if required_brake_force <= 0.0:
                safe_brake_level = brake_rl_level
                intervention = InterventionType.NONE
            else:
                min_brake_level = required_brake_force / p.brake_gain

                if brake_rl_level >= min_brake_level:
                    safe_brake_level = brake_rl_level
                    intervention = InterventionType.NONE
                else:
                    safe_brake_level = min_brake_level
                    intervention = InterventionType.BRAKE_ENFORCEMENT

        safe_throttle_percent = float(
            np.clip(safe_throttle_percent, 0.0, p.max_throttle_percent)
        )

        safe_brake_level = float(
            np.clip(safe_brake_level, 0.0, p.max_brake_level)
        )

        safe_action = np.array(
            [
                safe_throttle_percent / p.max_throttle_percent,
                safe_brake_level / p.max_brake_level,
            ],
            dtype=float,
        )

        original_action = np.array(
            [cbf_input.throttle_rl, cbf_input.brake_rl],
            dtype=float,
        )

        action_difference_l1 = float(np.linalg.norm(original_action - safe_action, ord=1))

        if intervention != InterventionType.NONE:
            logger.debug(
                "MO-CBF intervention=%s | RL=(%.3f, %.3f), SAFE=(%.3f, %.3f), "
                "h=%.3f, tau_max=%.3f, required_brake=%.3f",
                intervention.value,
                cbf_input.throttle_rl,
                cbf_input.brake_rl,
                safe_action[0],
                safe_action[1],
                h_value,
                torque_threshold,
                required_brake_force,
            )

        return CBFOutput(
            safe_action=safe_action,
            safe_throttle=safe_action[0],
            safe_brake=safe_action[1],
            intervention_type=intervention,
            barrier_value=h_value,
            torque_threshold=torque_threshold,
            required_brake_force=required_brake_force,
            action_difference_l1=action_difference_l1,
        )


def apply_mocbf(
    throttle_rl: float,
    brake_rl: float,
    leader_distance: float,
    ego_speed: float,
    leader_speed: float,
    current_rpm: float,
    vehicle_params: Optional[VehicleParameters] = None,
) -> np.ndarray:
    """
    Convenience wrapper for using MO-CBF like the original DoubleCBF class.

    Returns only the normalized safe action:

        [safe_throttle, safe_brake]
    """

    cbf = MOCBF(vehicle_params=vehicle_params)

    cbf_input = CBFInput(
        throttle_rl=throttle_rl,
        brake_rl=brake_rl,
        leader_distance=leader_distance,
        ego_speed=ego_speed,
        leader_speed=leader_speed,
        current_rpm=current_rpm,
    )

    return cbf.filter_action(cbf_input).safe_action


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    cbf = MOCBF()

    example_input = CBFInput(
        throttle_rl=0.8,
        brake_rl=0.0,
        leader_distance=15.0,
        ego_speed=12.0,
        leader_speed=8.0,
        current_rpm=2500.0,
    )

    output = cbf.filter_action(example_input)

    print("Original action:", [example_input.throttle_rl, example_input.brake_rl])
    print("Safe action:", output.safe_action)
    print("Intervention:", output.intervention_type.value)
    print("Barrier value h(x):", output.barrier_value)
    print("Torque threshold:", output.torque_threshold)
    print("Required brake force:", output.required_brake_force)
    print("Action difference L1:", output.action_difference_l1)
