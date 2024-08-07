"""
Debugging script: Runs trained model on training data to sanity check the inference pipeline.

Usage:
    cd droid_policy_learning

    python robomimic/scripts/evaluate_policy_on_data.py -l --ckpt_path /iris/u/moojink/prismatic-dev/droid_dp_runs/droid/im/diffusion_policy/04-25-None/bz_128_noise_samples_8_sample_weights_1_dataset_names_mjk_panda_4_cams_static_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/20240425183333/models/model_epoch_5000.pth
    python robomimic/scripts/evaluate_policy_on_data.py -l --ckpt_path /iris/u/moojink/prismatic-dev/droid_dp_runs/libero/im/diffusion_policy/07-20-None/bz_128_noise_samples_8_sample_weights_1_dataset_names_libero_spatial_cams_image_ldkeys_proprio-lang_visenc_VisualCore_fuser_None/20240721002112/models/model_epoch_50.pth
"""
import argparse
import os
import json
import numpy as np
import time
import torch
from copy import deepcopy

from droid.calibration.calibration_utils import load_calibration_info
from droid.camera_utils.info import camera_type_dict
from droid.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from droid.controllers.oculus_controller import VRPolicy
from droid.evaluation.policy_wrapper import PolicyWrapperRobomimic
from droid.robot_env import RobotEnv
from droid.user_interface.data_collector import DataCollecter
from droid.user_interface.gui import RobotGUI

import robomimic.utils.action_utils as AcUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils

import cv2

from torch.utils.data import DataLoader
import tensorflow as tf

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.action_utils as ActionUtils
from robomimic.utils.dataset import action_stats_to_normalization_stats
from robomimic.utils.rlds_utils import (
    droid_dataset_transform_abs, 
    droid_dataset_transform_rel, 
    libero_dataset_transform, 
    libero_dataset_transform_abs, 
    robomimic_transform,
    robomimic_transform_libero,
    DROID_TO_RLDS_OBS_KEY_MAP, 
    DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP, 
    TorchRLDSDataset
)

from octo.data.dataset import make_dataset_from_rlds, make_interleaved_dataset
from octo.data.utils.data_utils import combine_dataset_statistics
from octo.utils.spec import ModuleSpec

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def eval_launcher(variant, run_id, exp_id):
    """(For debugging) Launches eval of trained model on training data."""

    ###################################################################################################
    # Load policy
    ###################################################################################################

    # Get directory
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # Prepare log directory
    variant["exp_name"] = os.path.join(variant["exp_name"], "run{0}/id{1}/".format(run_id, exp_id))
    log_dir = os.path.join(dir_path, "./evaluation_logs", variant["exp_name"])

    # Set random seeds
    torch.manual_seed(variant["seed"])
    np.random.seed(variant["seed"])

    # Set compute mode
    use_gpu = variant.get("use_gpu", False)
    torch.device("cuda:0" if use_gpu else "cpu")

    ckpt_path = variant["ckpt_path"]

    # Load config
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    ckpt_dict = FileUtils.maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
    config = json.loads(ckpt_dict["config"])

    # Get input image size for policy
    for obs_key in ckpt_dict["shape_metadata"]["all_shapes"].keys():
        if "droid" in config['experiment']['name']:
            if 'static_image' in obs_key:
                imsize = max(ckpt_dict["shape_metadata"]["all_shapes"][obs_key])
                break
        elif "libero" in config['experiment']['name']:
            if 'image' in obs_key:
                imsize = max(ckpt_dict["shape_metadata"]["all_shapes"][obs_key])
                break
        else:
            raise ValueError("Unexpected experiment name found! Expecting DROID or LIBERO experiment.")


    # Load policy
    ckpt_dict["config"] = json.dumps(config)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_dict=ckpt_dict, device=device, verbose=True)
    policy.goal_mode = config["train"]["goal_mode"]
    policy.eval_mode = True

    # Determine the action space (relative or absolute)
    action_keys = config["train"]["action_keys"]
    if "action/rel_pos" in action_keys:
        action_space = "cartesian_velocity"
        for k in action_keys:
            assert not k.startswith("action/abs_")
    elif "action/abs_pos" in action_keys:
        action_space = "cartesian_position"
        for k in action_keys:
            assert not k.startswith("action/rel_")
    else:
        raise ValueError

    # Determine the action space for the gripper
    if "action/gripper_velocity" in action_keys:
        gripper_action_space = "velocity"
    elif "action/gripper_position" in action_keys:
        gripper_action_space = "position"
    else:
        raise ValueError

    # Prepare policy wrapper
    if "droid" in config['experiment']['name']:
        data_processing_kwargs = dict(
            timestep_filtering_kwargs=dict(
                action_space=action_space,
                gripper_action_space=gripper_action_space,
                robot_state_keys=["cartesian_position", "gripper_position", "joint_positions"],
            ),
            image_transform_kwargs=dict(
                remove_alpha=True,
                bgr_to_rgb=True,
                to_tensor=True,
                augment=False,  # because the DP test-time logic already handles augmentations
            ),
        )
    elif "libero" in config['experiment']['name']:
        data_processing_kwargs = dict(
            timestep_filtering_kwargs=dict(
                action_space=action_space,
                gripper_action_space=gripper_action_space,
                robot_state_keys=["state"],
            ),
            image_transform_kwargs=dict(
                remove_alpha=True,
                bgr_to_rgb=True,
                to_tensor=True,
                augment=False,  # because the DP test-time logic already handles augmentations
            ),
        )
    else:
        raise ValueError("Unexpected experiment name found! Expecting DROID or LIBERO experiment.")
    timestep_filtering_kwargs = data_processing_kwargs.get("timestep_filtering_kwargs", {})
    image_transform_kwargs = data_processing_kwargs.get("image_transform_kwargs", {})

    policy_data_processing_kwargs = {}
    policy_timestep_filtering_kwargs = policy_data_processing_kwargs.get("timestep_filtering_kwargs", {})
    policy_image_transform_kwargs = policy_data_processing_kwargs.get("image_transform_kwargs", {})

    policy_timestep_filtering_kwargs.update(timestep_filtering_kwargs)
    policy_image_transform_kwargs.update(image_transform_kwargs)

    fs = config["train"]["frame_stack"]  # Number of frames to stack

    wrapped_policy = PolicyWrapperRobomimic(
        policy=policy,
        timestep_filtering_kwargs=policy_timestep_filtering_kwargs,
        image_transform_kwargs=policy_image_transform_kwargs,
        frame_stack=fs,
        eval_mode=True,
    )

    camera_kwargs = dict(
        hand_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
        static_camera=dict(image=True, concatenate_images=False, resolution=(imsize, imsize), resize_func="cv2"),
    )
    
    policy_camera_kwargs = {}
    policy_camera_kwargs.update(camera_kwargs)

    camera_reader = MultiCameraWrapper(camera_kwargs)

    ###################################################################################################
    # Load training dataloader
    ###################################################################################################

    env_meta = {}
    obs_normalization_stats = None

    # (For RLDS datasets) Disable TensorFlow's GPU utilization
    tf.config.set_visible_devices([], "GPU")

    # Get RGB observation modalities
    obs_modalities = config["observation"]["modalities"]["obs"]["rgb"]
    assert(len(obs_modalities) == 1), "Only expecting observations from one camera!"

    # Get action dim, configs, and mask
    ac_dim = sum([ac_comp[1] for ac_comp in config["train"]["action_shapes"]])
    action_config = config["train"]["action_config"]
    action_mask = [True] * ac_dim

    # Set base dataset kwargs
    if "droid" in config['experiment']['name']:
        if "action/rel_pos" in config["train"]["action_keys"]:
            assert "action/abs_pos" not in config["train"]["action_keys"]
            dataset_transform = droid_dataset_transform_rel
        else:
            dataset_transform = droid_dataset_transform_abs
        state_obs_keys = [DROID_TO_RLDS_LOW_DIM_OBS_KEY_MAP[obs_key] for obs_key in config["observation"]["modalities"]["obs"]["low_dim"]]
    elif "libero" in config['experiment']['name']:
        if "action/rel_pos" in config["train"]["action_keys"]:
            assert "action/abs_pos" not in config["train"]["action_keys"]
            dataset_transform = libero_dataset_transform
        else:
            dataset_transform = libero_dataset_transform_abs
        state_obs_keys = config["observation"]["modalities"]["obs"]["low_dim"]
    else:
        raise ValueError("Unexpected experiment name found! Expecting DROID or LIBERO experiment.")
    BASE_DATASET_KWARGS = {
        "data_dir": config["train"]["data_path"],
        "image_obs_keys": {"primary": obs_modalities[0], "secondary": None},
        "state_obs_keys": state_obs_keys,
        "language_key": "language_instruction",
        "norm_skip_keys": ["proprio"],
        "action_proprio_normalization_type": "bounds",
        "absolute_action_mask": action_mask,
        "action_normalization_mask": action_mask,
        "standardize_fn": dataset_transform,
        }

    # Filter out failure episodes if applicable.
    dataset_names = config["train"]["dataset_names"]
    filter_functions = [[ModuleSpec.create(
                            "robomimic.utils.rlds_utils:filter_success"
                            )] if d_name == "droid" else [] \
                        for d_name in dataset_names]

    # Set dataset kwargs list.
    dataset_kwargs_list = [
        {"name": d_name, "filter_functions": f_functions, **BASE_DATASET_KWARGS} for d_name, f_functions in zip(dataset_names, filter_functions)
    ]

    # Compute combined normalization stats.
    combined_dataset_statistics = combine_dataset_statistics(
        [make_dataset_from_rlds(**dataset_kwargs, train=True)[1] for dataset_kwargs in dataset_kwargs_list]
    )

    # Create dataset.
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        config["train"]["sample_weights"],
        train=True,
        shuffle_buffer_size=config["train"]["shuffle_buffer_size"],
        batch_size=None,  # batching will be handled in PyTorch Dataloader object
        balance_weights=False,
        dataset_statistics=combined_dataset_statistics,
        traj_transform_kwargs=dict(
            # NOTE(Ashwin): window_size and future_action_window_size may break if 
            # not using diffusion policy
            window_size=config["algo"]["horizon"]["observation_horizon"],
            future_action_window_size=config["algo"]["horizon"]["prediction_horizon"]-1,  # -1 because horizon = current action (+1) + future (H-1) actions
            subsample_length=config["train"]["subsample_length"],
            skip_unlabeled=True,    # skip all trajectories without language
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs=dict(
            ),
            resize_size=dict(
                primary=config["observation"]["image_dim"],
                secondary=config["observation"]["image_dim"],
            ),
            num_parallel_calls=config["train"]["num_parallel_calls"],
        ),
        traj_transform_threads=config["train"]["traj_transform_threads"],
        traj_read_threads=config["train"]["traj_read_threads"],
    )
    # Note: If we have separated statistics for multiple datasets, use the first one (assumed to be DROID)
    # Otherwise, use the combined dataset statistics.
    rlds_dataset_stats = dataset.dataset_statistics[0] if isinstance(dataset.dataset_statistics, list) else dataset.dataset_statistics
    num_transitions = rlds_dataset_stats["num_transitions"][0].item()
    action_stats = ActionUtils.get_action_stats_dict(rlds_dataset_stats["action"], config["train"]["action_keys"], config["train"]["action_shapes"])
    action_normalization_stats = action_stats_to_normalization_stats(action_stats, action_config)
    if "droid" in config['experiment']['name']:
        dataset = dataset.map(robomimic_transform, num_parallel_calls=config["train"]["traj_transform_threads"])
    elif "libero" in config['experiment']['name']:
        dataset = dataset.map(robomimic_transform_libero, num_parallel_calls=config["train"]["traj_transform_threads"])
    else:
        raise ValueError("Unexpected experiment name found! Expecting DROID or LIBERO experiment.")

    # Create PyTorch Dataset and DataLoader.
    pytorch_dataset = TorchRLDSDataset(dataset)
    train_loader = DataLoader(
        pytorch_dataset,
        batch_size=1,
        num_workers=0,  # important to keep this to 0 so PyTorch does not mess with the parallelism
    )

    ###################################################################################################
    # Run inference on training batches.
    ###################################################################################################

    total_l1_loss = 0.
    total_l2_loss = 0.
    cnt = 0

    for batch in train_loader:
        assert len(batch['obs']['raw_language']) == 1, "This eval script only supports batch size 1 because it requires writing the language instruction to a file to specify the current task!"

        # SUPER HACKY: Write the language instruction to a file, since DP expects to see this file during inference
        if not os.path.exists("eval_params"):
            os.makedirs("eval_params")
        with open("eval_params/lang_command.txt", "w") as file:
            task_label = batch['obs']['raw_language'][0].decode('utf-8')
            file.write(task_label)

        curr_actions = []
        wrapped_policy.fs_wrapper.reset()

        for i in range(config["algo"]["horizon"]["observation_horizon"]):
            obs_dict = {"timestamp": {}}

            #############################################################################################################################
            # [Start] Setup mandatory camera reader stuff...
            #############################################################################################################################
            # Robot State #
            # batch['obs']['robot_state/cartesian_position'][:,:,3] = -1 * batch['obs']['robot_state/cartesian_position'][:,:,3] # TODO
            # batch['obs']['robot_state/cartesian_position'] = -1 * batch['obs']['robot_state/cartesian_position'] # TODO
            if "droid" in config['experiment']['name']:
                state_dict = {
                    "cartesian_position": batch['obs']['robot_state/cartesian_position'][0,i],
                    "gripper_position": batch['obs']['robot_state/gripper_position'][0,i].item(), # should be single float, not array
                    "joint_positions": [0,0,0,0,0,0,0],
                    "joint_velocities": [0,0,0,0,0,0,0],
                    "joint_torques_computed": [0,0,0,0,0,0,0],
                    "prev_joint_torques_computed": [0,0,0,0,0,0,0],
                    "prev_joint_torques_computed_safened": [0,0,0,0,0,0,0],
                    "motor_torques_measured": [0,0,0,0,0,0,0],
                    "prev_controller_latency_ms": [0,0,0,0,0,0,0],
                    "prev_command_successful": True,
                }
            elif "libero" in config['experiment']['name']:
                state_dict = {
                    # "cartesian_position": batch['obs']['state'][0,i,:6],
                    # "gripper_position": batch['obs']['state'][0,i,-1].item(), # should be single float, not array
                    "state": batch['obs']['state'][0,i],
                    "joint_positions": [0,0,0,0,0,0,0],
                    "joint_velocities": [0,0,0,0,0,0,0],
                    "joint_torques_computed": [0,0,0,0,0,0,0],
                    "prev_joint_torques_computed": [0,0,0,0,0,0,0],
                    "prev_joint_torques_computed_safened": [0,0,0,0,0,0,0],
                    "motor_torques_measured": [0,0,0,0,0,0,0],
                    "prev_controller_latency_ms": [0,0,0,0,0,0,0],
                    "prev_command_successful": True,
                }
            else:
                raise ValueError("Unexpected experiment name found! Expecting DROID or LIBERO experiment.")
            timestamp_dict = {
                "robot_timestamp_seconds": time.time(),
                "robot_timestamp_nanos": time.time(),
            }
            obs_dict["robot_state"] = state_dict
            obs_dict["timestamp"]["robot_state"] = timestamp_dict

            # Camera Readings #
            hand_camera_id = '138422074005'
            static_camera_id = '140122076178'
            if "droid" in config['experiment']['name']:
                camera_obs = {
                    "image": {
                        hand_camera_id: np.zeros((128,128,3), dtype=np.uint8),
                        static_camera_id: batch['obs']['static_image'][0,i].numpy(),
                    }
                }
            elif "libero" in config['experiment']['name']:
                camera_obs = {
                    "image": {
                        hand_camera_id: np.zeros((128,128,3), dtype=np.uint8),
                        static_camera_id: batch['obs']['image'][0,i].numpy(),
                    }
                }
            else:
                raise ValueError("Unexpected experiment name found! Expecting DROID or LIBERO experiment.")
            camera_timestamp = {}
            obs_dict.update(camera_obs)
            obs_dict["timestamp"]["cameras"] = camera_timestamp

            # Camera Info #
            obs_dict["camera_type"] = deepcopy(camera_type_dict)
            # Adjust gripper camere by current pose
            calibration_dict = load_calibration_info()
            extrinsics = deepcopy(calibration_dict)
            for cam_id in calibration_dict:
                if hand_camera_id not in cam_id:
                    continue
                if "droid" in config['experiment']['name']:
                    gripper_pose = state_dict["cartesian_position"]
                elif "libero" in config['experiment']['name']:
                    gripper_pose = state_dict["state"][:6]
                else:
                    raise ValueError("Unexpected experiment name found! Expecting DROID or LIBERO experiment.")
                extrinsics[cam_id + "_gripper_offset"] = extrinsics[cam_id]
                extrinsics[cam_id] = extrinsics[cam_id]
            obs_dict["camera_extrinsics"] = extrinsics

            intrinsics = {}
            for cam in camera_reader.camera_dict.values():
                cam_intr_info = cam.get_intrinsics()
                for (full_cam_id, info) in cam_intr_info.items():
                    intrinsics[full_cam_id] = info["cameraMatrix"]
            obs_dict["camera_intrinsics"] = intrinsics

            #############################################################################################################################
            # [End] Setup mandatory camera reader stuff...
            #############################################################################################################################

            # Policy forward #
            Ta = config["algo"]["horizon"]["action_horizon"]
            action = wrapped_policy.forward(obs_dict)
            curr_actions.append(action)

            # Flush out the rest of the action chunk.
            if i == config["algo"]["horizon"]["observation_horizon"] - 1:
                while len(curr_actions) < Ta:
                    action = wrapped_policy.forward(obs_dict)
                    curr_actions.append(action)

            if len(curr_actions) == Ta:
                predicted_actions = np.stack(curr_actions)
                ground_truth_actions = batch['actions'][0,:Ta].cpu().numpy()
                # Un-normalize ground truth actions.
                action_keys = config["train"]["action_keys"]
                action_shapes = {k: action_normalization_stats[k]["offset"].shape[1:] for k in action_normalization_stats}
                ground_truth_actions_dict = AcUtils.vector_to_action_dict(ground_truth_actions, action_shapes=action_shapes, action_keys=action_keys)
                ground_truth_actions_dict = ObsUtils.unnormalize_dict(ground_truth_actions_dict, normalization_stats=action_normalization_stats)
                ground_truth_actions = AcUtils.action_dict_to_vector(ground_truth_actions_dict, action_keys=action_keys)

                # (If applicable) Convert 6-D rotations to 3-D Eulers in the ground truth actions.
                if "action/abs_rot_6d" in action_keys:
                    ground_truth_actions_dict = AcUtils.vector_to_action_dict(ground_truth_actions, action_shapes=action_shapes, action_keys=action_keys)
                    rot_6d = torch.from_numpy(ground_truth_actions[:,3:9])
                    rot = TorchUtils.rot_6d_to_euler_angles(rot_6d=rot_6d).squeeze().numpy()
                    ground_truth_actions_dict["action/abs_rot_6d"] = rot
                    ground_truth_actions = AcUtils.action_dict_to_vector(ground_truth_actions_dict, action_keys=action_keys)

                print(f"cnt: {cnt}")
                print(f"ground_truth_actions:\n{ground_truth_actions}")
                print(f"predicted_actions:\n{predicted_actions}")
                l1_loss = torch.nn.functional.l1_loss(torch.Tensor(predicted_actions), torch.Tensor(ground_truth_actions), reduction='mean').item()
                l2_loss = torch.nn.functional.mse_loss(torch.Tensor(predicted_actions), torch.Tensor(ground_truth_actions), reduction='mean').item()
                print(f"l1_loss: {l1_loss:.3f}")
                print(f"l2_loss: {l2_loss:.3f}")
                total_l1_loss += l1_loss
                total_l2_loss += l2_loss
                cnt += 1
        if cnt == 200:
            break

    avg_l1_loss = total_l1_loss / cnt
    avg_l2_loss = total_l2_loss / cnt
    print(f"============= Summary =============")
    print(f"avg_l1_loss: {avg_l1_loss}")
    print(f"avg_l2_loss: {avg_l2_loss}")
    print(f"cnt: {cnt}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--lang_cond', action='store_true')
    parser.add_argument('-c', '--ckpt_path', type=str, default=None, 
        help='Path to Pytorch checkpoint (.pth) corresponding to the policy you want to evaluate.')
    args = parser.parse_args()

    variant = dict(
        exp_name="policy_test",
        save_data=False,
        use_gpu=True,
        seed=0,
        policy_logdir="test",
        task="",
        layout_id=None,
        model_id=50,
        camera_kwargs=dict(),
        data_processing_kwargs=dict(
            timestep_filtering_kwargs=dict(),
            image_transform_kwargs=dict(),
        ),
        ckpt_path=args.ckpt_path,
    )

    print("Evaluating Policy")
    eval_launcher(variant, run_id=1, exp_id=0)
