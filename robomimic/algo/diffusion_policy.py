"""
Implementation of Diffusion Policy https://diffusion-policy.cs.columbia.edu/ by Cheng Chi
"""
from typing import Callable, Union
import math
from collections import OrderedDict, deque
from packaging.version import parse as parse_version
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# requires diffusers==0.11.1
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.training_utils import EMAModel

import robomimic.models.obs_nets as ObsNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.algo import register_algo_factory_func, PolicyAlgo

import random
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import os


from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
lang_model = AutoModel.from_pretrained("distilbert-base-uncased", torch_dtype=torch.float16)
lang_model.to('cuda')


# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP

import cv2
import copy


@register_algo_factory_func("diffusion_policy")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the BC algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if algo_config.unet.enabled:
        return DiffusionPolicyUNet, {}
    elif algo_config.transformer.enabled:
        raise NotImplementedError()
    else:
        raise RuntimeError()

class DiffusionPolicyUNet(PolicyAlgo):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        # set up different observation groups for @MIMO_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)
        encoder_kwargs = ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder)
        
        obs_encoder = ObsNets.ObservationGroupEncoder(
            observation_group_shapes=observation_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # IMPORTANT!
        # replace all BatchNorm with GroupNorm to work with EMA
        # performance will tank if you forget to do this!
        obs_encoder = replace_bn_with_gn(obs_encoder)
        
        obs_dim = obs_encoder.output_shape()[0]

        # create network object
        noise_pred_net = ConditionalUnet1D(
            input_dim=self.ac_dim,
            global_cond_dim=obs_dim*self.algo_config.horizon.observation_horizon
        )

        # the final arch has 2 parts
        nets = nn.ModuleDict({
            'policy': nn.ModuleDict({
                'obs_encoder': torch.nn.parallel.DataParallel(obs_encoder, device_ids=list(range(0,torch.cuda.device_count()))),
                'noise_pred_net': torch.nn.parallel.DataParallel(noise_pred_net, device_ids=list(range(0,torch.cuda.device_count())))
            })
        })

        nets = nets.float().to(self.device)
        
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        
        # setup EMA
        ema = None
        if self.algo_config.ema.enabled:
            ema = EMAModel(model=nets, power=self.algo_config.ema.power)
                
        # set attrs
        self.nets = nets
        self.noise_scheduler = noise_scheduler
        self.ema = ema
        self.action_check_done = False
        self.obs_queue = None
        self.action_queue = None

        # Set observation, action, and prediction horizon lengths. Set action dim as well.
        self.To = self.algo_config.horizon.observation_horizon
        self.Ta = self.algo_config.horizon.action_horizon
        self.Tp = self.algo_config.horizon.prediction_horizon

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training 
        """

        input_batch = dict()

        ## Semi-hacky fix which does the filtering for raw language which is just a list of lists of strings
        input_batch["obs"] = {k: batch["obs"][k][:, :self.To, :] for k in batch["obs"] if "raw" not in k }
        if "lang_fixed/language_raw" in batch["obs"].keys():
            str_ls = list(batch['obs']['lang_fixed/language_raw'][0])
            input_batch["obs"]["lang_fixed/language_raw"] = [str_ls] * self.To

        with torch.no_grad():
            if "raw_language" in batch["obs"].keys():
                raw_lang_strings = [byte_string.decode('utf-8') for byte_string in batch["obs"]['raw_language']]
                encoded_input = tokenizer(raw_lang_strings, padding=True, truncation=True, return_tensors='pt').to('cuda')
                outputs = lang_model(**encoded_input)
                encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(1).repeat(1, self.To, 1)
                input_batch["obs"]["lang_fixed/language_distilbert"] = encoded_lang.type(torch.float32)

        input_batch["actions"] = batch["actions"][:, :self.Tp, :]
        
        # check if actions are normalized to [-1,1]
        if not self.action_check_done:
            actions = input_batch["actions"]
            in_range = (-1 <= actions) & (actions <= 1)
            all_in_range = torch.all(in_range).item()
            if not all_in_range:
                raise ValueError('"actions" must be in range [-1,1] for Diffusion Policy! Check if hdf5_normalize_action is enabled.')
            self.action_check_done = True

        ## LOGGING HOW MANY NANs there are
        # bz = input_batch["actions"].shape[0]
        # nanamt = torch.BoolTensor([False] * bz)
        # for key in input_batch["obs"]:
        #     if key == "pad_mask":
        #         continue
        #     nanamt = torch.logical_or(nanamt, torch.isnan(input_batch["obs"][key].reshape(bz, -1).mean(1)))
        # print(nanamt.float().mean())

        for key in input_batch["obs"]:
            input_batch["obs"][key] = torch.nan_to_num(input_batch["obs"][key])
        input_batch["actions"] = torch.nan_to_num(input_batch["actions"])
        
        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def train_on_batch(self, batch, epoch, validate=False, get_actions_l1_loss=False):
        """
        Training on a single batch of data.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

            epoch (int): epoch number - required by some Algos that need
                to perform staged training and early stopping

            validate (bool): if True, don't perform any learning updates.

            get_actions_l1_loss (bool): if True, get L1 loss b/t ground-truth and predicted actions

        Returns:
            info (dict): dictionary of relevant inputs, outputs, and losses
                that might be relevant for logging
        """
        B = batch['actions'].shape[0]

        with TorchUtils.maybe_no_grad(no_grad=validate):
            info = super(DiffusionPolicyUNet, self).train_on_batch(batch, epoch, validate=validate)
            actions = batch['actions']
            
            # encode obs
            inputs = {
                'obs': batch["obs"],
            }
            for k in self.obs_shapes:
                ## Shape assertion does not apply to list of strings for raw language
                if "raw" in k:
                    continue
                # first two dimensions should be [B, T] for inputs
                assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k])

            obs_features = TensorUtils.time_distributed({"obs":inputs["obs"]}, self.nets['policy']['obs_encoder'], inputs_as_kwargs=True)
            assert obs_features.ndim == 3  # [B, T, D]
            obs_cond = obs_features.flatten(start_dim=1)

            num_noise_samples = self.algo_config.noise_samples

            # sample noise to add to actions
            noise = torch.randn([num_noise_samples] + list(actions.shape), device=self.device)
            
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, 
                (B,), device=self.device
            ).long()
            
            # add noise to the clean actions according to the noise magnitude at each diffusion iteration
            # (this is the forward diffusion process)
            noisy_actions = torch.cat([self.noise_scheduler.add_noise(
                            actions, noise[i], timesteps)
                            for i in range(len(noise))], dim=0)

            obs_cond = obs_cond.repeat(num_noise_samples, 1)
            timesteps = timesteps.repeat(num_noise_samples)

            # predict the noise residual
            noise_pred = self.nets['policy']['noise_pred_net'](
                noisy_actions, timesteps, global_cond=obs_cond)

            # L2 loss
            noise = noise.view(noise.size(0) * noise.size(1), *noise.size()[2:])
            loss = F.mse_loss(noise_pred, noise)

            # logging
            losses = {
                'l2_loss': loss
            }

            # Also record L1 difference between ground-truth and predicted actions
            if get_actions_l1_loss:
                with torch.no_grad():
                    # Get ground-truth action trajectories
                    ground_truth_actions = actions
                    # Get full predicted action trajectories (by disabling receding horizon control)
                    predicted_actions = self._get_action_trajectory(obs_dict=batch["obs"], receding_horizon_control=False)  # (1, Ta, ac_dim)
                    # Get L1 loss
                    l1_loss = F.l1_loss(predicted_actions, ground_truth_actions)
                    # Add L1 loss to loss dict
                    losses["actions_l1_loss"] = l1_loss

            info["losses"] = TensorUtils.detach(losses)

            if not validate:
                # gradient step
                policy_grad_norms = TorchUtils.backprop_for_loss(
                    net=self.nets,
                    optim=self.optimizers["policy"],
                    loss=loss,
                )
                
                # update Exponential Moving Average of the model weights
                if self.ema is not None:
                    self.ema.step(self.nets)
                
                step_info = {
                    'policy_grad_norms': policy_grad_norms
                }
                info.update(step_info)

        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(DiffusionPolicyUNet, self).log_info(info)
        log["Loss"] = info["losses"]["l2_loss"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        if "actions_l1_loss" in info["losses"]:
            log["Actions_L1_Loss"] = info["losses"]["actions_l1_loss"].item()
        return log

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        # setup inference queues
        obs_queue = deque(maxlen=self.To)
        action_queue = deque(maxlen=self.Ta)
        self.obs_queue = obs_queue
        self.action_queue = action_queue

    def get_action(self, obs_dict, goal_mode=None, eval_mode=False):
        """
        Get policy action.

        Args:
            obs_dict (dict): current observation [1, Do]
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor [1, Da]
        """

        # obs_dict: key: [1,D]

        if eval_mode:
            root_path = os.path.join(os. getcwd(), "eval_params")

            if goal_mode is not None:  # goal conditioning
                # Read in goal images
                from droid.misc.parameters import hand_camera_id, static_camera_id
                goal_hand_camera_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{hand_camera_id}.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)
                goal_static_camera_image = torch.FloatTensor((cv2.cvtColor(cv2.imread(os.path.join(root_path, f"{static_camera_id}.png")), cv2.COLOR_BGR2RGB) / 255.0)).cuda().permute(2, 0, 1).unsqueeze(0).repeat([1, 1, 1, 1]).unsqueeze(0)

                obs_dict['wrist_image'] = torch.cat([obs_dict['wrist_image'], goal_hand_camera_image.repeat(1, self.To, 1, 1, 1)], dim=2)
                obs_dict['static_image'] = torch.cat([obs_dict['static_image'], goal_static_camera_image.repeat(1, self.To, 1, 1, 1)], dim=2)
            # Note: currently assumes that you are never doing both goal and language conditioning
            else:  # language conditioning
                # Get current language instruction
                raw_lang = obs_dict["task_label"]
                # Feed language instruction through language model.
                tokenized_lang = tokenizer(raw_lang, return_tensors='pt').to('cuda')
                outputs = lang_model(**tokenized_lang)  # (1, seq_len, transformer_dim)
                # To get language embedding, sum over the hidden states for the tokens in the sequence, and then replicate
                # the result `To` == `observation_horizon` times.
                encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0).repeat(self.To, 1).unsqueeze(0) # (1, To, transformer_dim)
                obs_dict["lang_fixed/language_distilbert"] = encoded_lang.type(torch.float32)

        ###############################

        # If there are no actions left to execute, run inference to predict another action chunk.
        if len(self.action_queue) == 0:
            # print(f"Predicting new action chunk...")
            # Predict action chunk.
            action_sequence = self._get_action_trajectory(obs_dict=obs_dict)  # (1, Ta, ac_dim)
            # Store action chunk in queue.
            self.action_queue.extend(action_sequence[0])

        # Execute first action in queue.
        action = self.action_queue.popleft()

        action = action.unsqueeze(0)  # (1, ac_dim)
        return action

    def _get_action_trajectory(self, obs_dict, receding_horizon_control=True):
        # Set number of diffusion inference timesteps based on DDPM or DDIM.
        if self.algo_config.ddpm.enabled is True:
            num_inference_timesteps = self.algo_config.ddpm.num_inference_timesteps
        elif self.algo_config.ddim.enabled is True:
            num_inference_timesteps = self.algo_config.ddim.num_inference_timesteps
        else:
            raise ValueError

        # Select network.
        nets = self.nets
        if self.ema is not None:
            nets = self.ema.averaged_model
        
        # Get observations.
        inputs = {
            'obs': obs_dict,
        }

        # Remove the task label from the observations dict if we already have the embedding for it
        # to prevent runtime errors later
        if "lang_fixed/language_distilbert" in inputs["obs"] and "task_label" in inputs["obs"]:
            inputs["obs"].pop("task_label")

        # Check that observations have the right shape.
        for k in self.obs_shapes:
            # Skip language strings.
            if "raw" in k:
                continue
            assert inputs['obs'][k].ndim - 2 == len(self.obs_shapes[k]), "First two dimensions should be [B, To] for inputs!"

        # Extract only the first `self.To` steps of observations.
        inputs["obs"] = {k: inputs["obs"][k][:, :self.To, :] for k in inputs["obs"]}

        # Encode observations, which are used to condition the reverse diffusion process.
        obs_features = TensorUtils.time_distributed({"obs":inputs["obs"]}, nets['policy']['obs_encoder'].module, inputs_as_kwargs=True)
        assert obs_features.ndim == 3  # [B, To, D]
        B = obs_features.shape[0]

        # Reshape observations to (B, observation_horizon * observation_dim) == (B, To * D).
        obs_cond = obs_features.flatten(start_dim=1)

        # Initialize action from randomly sampled Guassian noise of shape (B, Tp, ac_dim).
        noisy_action = torch.randn((B, self.Tp, self.ac_dim), device=self.device)
        
        # Initialize noise scheduler.
        self.noise_scheduler.set_timesteps(num_inference_timesteps)

        # Predict action via reverse diffusion.
        for k in self.noise_scheduler.timesteps:
            # Predict noise.
            noise_pred = nets['policy']['noise_pred_net'].module(
                sample=noisy_action,
                timestep=k,
                global_cond=obs_cond
            )
            # Apply inverse diffusion step (remove noise).
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=noisy_action
            ).prev_sample

        if receding_horizon_control:
            # Receding horizon control: Extract only the first `Ta` == `action_horizon` steps of actions.
            action = noisy_action[:,:self.Ta]  # (B, Ta, ac_dim)
        else:
            action = noisy_action
        return action

    def serialize(self):
        """
        Get dictionary of current model parameters.
        """
        return {
            "nets": self.nets.state_dict(),
            "ema": self.ema.averaged_model.state_dict() if self.ema is not None else None,
        }

    def deserialize(self, model_dict):
        """
        Load model from a checkpoint.

        Args:
            model_dict (dict): a dictionary saved by self.serialize() that contains
                the same keys as @self.network_classes
        """
        self.nets.load_state_dict(model_dict["nets"])
        if model_dict.get("ema", None) is not None:
            self.ema.averaged_model.load_state_dict(model_dict["ema"])



# =================== Vision Encoder Utils =====================
def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    if parse_version(torch.__version__) < parse_version('1.9.0'):
        raise ImportError('This function requires pytorch >= 1.9.0')

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module, 
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group, 
            num_channels=x.num_features)
    )
    return root_module

# =================== UNet for Diffusion ==============

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        global_cond_dim,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM 
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level. 
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            global_cond=None):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # If there is only 1 action (i.e. no action chunk), do not down-/up-sample.
        if sample.shape[-1] == 1:
            resize_sample = False
        else:
            resize_sample = True

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            if resize_sample:
                x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            if resize_sample:
                x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
