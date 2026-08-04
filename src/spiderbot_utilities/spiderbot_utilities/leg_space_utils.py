"""Utility functions for the leg space of a Spiderbot."""

import mujoco

import numpy as np


def sample_reachable_leg_space(leg, step_degree=15.0):
    """Sample reachable leg space."""
    step_radians = np.radians(step_degree)
    joint_ranges = leg.model.jnt_range[leg.leg_joint_ids]

    # Create a list of arrays of each joint angle
    joint_samples = [
        # Include half a step so the final step is the end of the range
        np.arange(joint_range[0], joint_range[1] + (step_radians / 2.0),
                  step_radians)
        for joint_range in joint_ranges
    ]

    # Cartesian product of all joints
    q_samples = (
        np.stack(np.meshgrid(*joint_samples, indexing='ij'), axis=-1)
        .reshape(-1, 3)
    )

    num_samples = len(q_samples)
    points = np.zeros((num_samples, 3))

    initial_qpos = leg.data.qpos[leg.leg_qpos_adrs].copy()

    for i, q in enumerate(q_samples):
        leg.data.qpos[leg.leg_qpos_adrs] = q
        mujoco.mj_fwdPosition(leg.model, leg.data)

        claw_pos = leg.data.site(leg.claw_tip_site_id).xpos.copy()

        coxa_pos = leg.data.site(leg.leg_base_site_id).xpos
        r_coxa = leg.data.site(leg.leg_base_site_id).xmat.reshape(3, 3)
        claw_pos = r_coxa.T @ (claw_pos - coxa_pos)

        points[i] = claw_pos

    leg.data.qpos[leg.leg_qpos_adrs] = initial_qpos
    mujoco.mj_fwdPosition(leg.model, leg.data)

    return points


def draw_leg_space_in_mujoco(spec, leg, leg_space_points):
    """Draw a set of points relative to the leg in world space."""
    subsampled = leg_space_points[::100]
    r_leg_base = leg.data.site(leg.leg_base_site_id).xmat.reshape(3, 3)
    leg_base_xyz = leg.data.site(leg.leg_base_site_id).xpos

    for index, point in enumerate(subsampled):
        point = leg_base_xyz + (r_leg_base @ point)
        spec.worldbody.add_site(
            name=f'{leg.id}_ws_{index}',
            pos=point,
            size=[0.003, 0.0, 0.0],
            rgba=[0.0, 0.8, 0.2, 0.3],
            type=mujoco.mjtGeom.mjGEOM_SPHERE
        )
