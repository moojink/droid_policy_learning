"""Episode transforms for different RLDS datasets to canonical dataset definition."""
from typing import Any, Dict

import numpy as np
import tensorflow as tf
import torch
import tensorflow_graphics.geometry.transformation as tfg

def filter_success(trajectory: dict[str, any]):
    # only keep trajectories that have "success" in the file path
    return tf.strings.regex_full_match(
        trajectory['traj_metadata']['episode_metadata']['file_path'][0],
        ".*/success/.*"
    )


def euler_to_rmat(euler):
    return tfg.rotation_matrix_3d.from_euler(euler)


def mat_to_rot6d(mat):
    r6 = mat[..., :2, :]
    r6_0, r6_1 = r6[..., 0, :], r6[..., 1, :]
    r6_flat = tf.concat([r6_0, r6_1], axis=-1)
    return r6_flat


def binarize_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > 0.95, actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: tf.cast(carry, tf.float32), lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True)


def droid_dataset_transform_abs(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # absolute cartesian control
    # every input feature is batched, ie has leading batch dimension
    T = trajectory["action_dict"]["cartesian_position"][:, :3]
    R = mat_to_rot6d(euler_to_rmat(trajectory["action_dict"]["cartesian_position"][:, 3:6]))
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            trajectory["action_dict"]["gripper_position"],
        ),
        axis=-1,
    )
    return trajectory


def droid_dataset_transform_rel(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # relative cartesian control
    # every input feature is batched, ie has leading batch dimension
    T = trajectory["action_dict"]["cartesian_velocity"][:, :3]
    R = trajectory["action_dict"]["cartesian_velocity"][:, 3:6]
    trajectory["action"] = tf.concat(
        (
            T,
            R,
            trajectory["action_dict"]["gripper_velocity"],
        ),
        axis=-1,
    )
    return trajectory


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state

    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, -2:],
        ),
        axis=-1,
    )
    return trajectory


def libero_dataset_transform_abs(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    dT = trajectory["action"][:, :3]
    T = trajectory["observation"]["state"][:, :3]
    T_prime = T + dT

    dR = euler_to_rmat(trajectory["action"][:, 3:6])
    R = euler_to_rmat(trajectory["observation"]["state"][:, 3:6])
    R_prime = mat_to_rot6d(dR @ R)

    dG = binarize_gripper_actions(trajectory["action"][:, -1])[:, None]
    G = trajectory["observation"]["state"][:, -2:-1]
    G_prime = G + dG

    trajectory["action"] = tf.concat(
        [
            T_prime,
            R_prime,
            G_prime
        ],
        axis=1,
    )

    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state

    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :6],
            trajectory["observation"]["state"][:, -2:],
        ),
        axis=-1,
    )
    return trajectory


def robomimic_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "static_image": trajectory["observation"]["image_primary"],
            "raw_language": trajectory["task"]["language_instruction"],
            "robot_state/cartesian_position": trajectory["observation"]["proprio"][..., :6],
            "robot_state/gripper_position": trajectory["observation"]["proprio"][..., -1:],
            "pad_mask": trajectory["observation"]["pad_mask"][..., None],
        },
        "actions": trajectory["action"][1:],
    }

def robomimic_transform_libero(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "obs": {
            "image": trajectory["observation"]["image_primary"],
            "raw_language": trajectory["task"]["language_instruction"],
            "state": trajectory["observation"]["proprio"],
        },
        "actions": trajectory["action"],
    }

DROID_TO_RLDS_OBS_KEY_MAP = {
    "static_image": "static_image",
}

DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP = {
    "robot_state/cartesian_position": "cartesian_position",
    "robot_state/gripper_position": "gripper_position",
}

class TorchRLDSDataset(torch.utils.data.IterableDataset):
    """Thin wrapper around RLDS dataset for use with PyTorch dataloaders."""

    def __init__(
        self,
        rlds_dataset,
        train=True,
    ):
        self._rlds_dataset = rlds_dataset
        self._is_train = train

    def __iter__(self):
        for sample in self._rlds_dataset.as_numpy_iterator():
            yield sample

    def __len__(self):
        lengths = np.array(
            [
                stats["num_transitions"]
                for stats in self._rlds_dataset.dataset_statistics
            ]
        )
        if hasattr(self._rlds_dataset, "sample_weights"):
            lengths *= np.array(self._rlds_dataset.sample_weights)
        total_len = lengths.sum()
        if self._is_train:
            return int(0.95 * total_len)
        else:
            return int(0.05 * total_len)

