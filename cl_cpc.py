# cl_cpc.py (FULL COPY-PASTE)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import encoder

LOG_FREQ = 25_000


# -------------------------
# Debug / safety helpers
# -------------------------
def _is_finite_tensor(x: torch.Tensor) -> bool:
    if not torch.is_tensor(x):
        return True
    if x.numel() == 0:
        return True
    return bool(torch.isfinite(x).all().item())


def _tensor_brief(x: torch.Tensor, name: str) -> str:
    if not torch.is_tensor(x):
        return f"{name}: <not a tensor>"
    try:
        return (
            f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"min={float(x.min().item()) if x.numel() else 'NA'} "
            f"max={float(x.max().item()) if x.numel() else 'NA'} "
            f"finite={_is_finite_tensor(x)}"
        )
    except Exception:
        return f"{name}: shape={tuple(x.shape)} dtype={x.dtype} device={x.device} finite=? (brief failed)"


def _assert_ok_tensor(x: torch.Tensor, name: str):
    if not torch.is_tensor(x):
        raise RuntimeError(f"[CURL_SAC][BAD] {name} is not a torch.Tensor")
    if x.numel() == 0:
        raise RuntimeError(f"[CURL_SAC][BAD] {name} is empty tensor (numel=0). { _tensor_brief(x, name) }")
    if not _is_finite_tensor(x):
        raise RuntimeError(f"[CURL_SAC][BAD] {name} has NaN/Inf. { _tensor_brief(x, name) }")


def _safe_cuda_sync():
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = encoder.CNNEncoder(
            obs_shape, encoder_feature_dim, num_layers, num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            noise = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)
        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return
        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)
        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class TransitionModel(nn.Module):
    def __init__(self, z_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim + action_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.apply(weight_init)

    def forward(self, z, action):
        x = torch.cat([z, action], dim=1)
        x = self.fc1(x)
        x = self.ln(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employs two Q-functions."""
    def __init__(self, obs_shape, action_shape, hidden_dim,
                 encoder_feature_dim, num_layers, num_filters):
        super().__init__()
        self.encoder = encoder.CNNEncoder(
            obs_shape, encoder_feature_dim, num_layers, num_filters, output_logits=True
        )
        self.Q1 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)
        self.Q2 = QFunction(self.encoder.feature_dim, action_shape[0], hidden_dim)

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        obs = self.encoder(obs, detach=detach_encoder)
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)
        self.outputs['q1'] = q1
        self.outputs['q2'] = q2
        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return
        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)
        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class CURL(nn.Module):
    def __init__(self, obs_shape, z_dim, critic, critic_target, output_type="continuous"):
        super(CURL, self).__init__()
        self.encoder = critic.encoder
        self.encoder_target = critic_target.encoder
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x, detach=False, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)
        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        # Expected: z_a [B, D], z_pos [B, D] => logits [B, B]
        # Use stable centering to avoid large values.
        Wz = torch.matmul(self.W, z_pos.T)       # [D, B]
        logits = torch.matmul(z_a, Wz)           # [B, B]
        logits = logits - torch.max(logits, 1, keepdim=True)[0]
        return logits


class CurlSacAgent(object):
    """CURL representation learning with SAC."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        augmentor,
        hidden_dim=1024,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=3e-4,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=3e-4,
        critic_beta=0.9,
        critic_tau=0.01,
        critic_target_update_freq=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.05,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        log_param_hist_imgs=False,
        detach_encoder=False,
        pixel_sac=False,
        predictive_cpc=True
    ):
        self.augmentor = augmentor
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.log_param_hist_imgs = log_param_hist_imgs
        self.image_shape = obs_shape[-2:]
        self.detach_encoder = detach_encoder
        self.pixel_sac = pixel_sac
        self.predictive_cpc = predictive_cpc

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature), dtype=torch.float32, device=device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -float(np.prod(action_shape))

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999))

        self.CURL = CURL(
            obs_shape, encoder_feature_dim,
            self.critic, self.critic_target,
            output_type='continuous'
        ).to(self.device)

        self.transition_model = TransitionModel(
            encoder_feature_dim, action_shape[0], hidden_dim
        ).to(self.device)
        self.W_pred = nn.Parameter(torch.rand(encoder_feature_dim, encoder_feature_dim, device=self.device))

        self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=encoder_lr)
        self.cpc_optimizer = torch.optim.Adam(self.CURL.parameters(), lr=encoder_lr)
        self.transition_optimizer = torch.optim.Adam(
            list(self.transition_model.parameters()) + [self.W_pred],
            lr=3e-4
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

        # ✅ FIX: CurlSacAgent now has train()/eval()
        self.train(True)
        # target network also in train mode by default
        self.critic_target.train(True)

    # -------------------------
    # Mode / alpha
    # -------------------------
    def train(self, training: bool = True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.CURL.train(training)
        self.transition_model.train(training)
        # target can stay in train mode; it's not updated by grads anyway
        self.critic_target.train(training)
        return self

    def eval(self):
        return self.train(False)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # -------------------------
    # Acting
    # -------------------------
    def select_action(self, obs):
        """Deterministic action (mu). obs is numpy [C,H,W]."""
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().numpy().flatten()

    def sample_action(self, obs):
        """Stochastic action (pi). obs is numpy [C,H,W]."""
        if obs.shape[-2:] != self.image_shape:
            obs = self.augmentor.evaluation_augmentation(obs)
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().numpy().flatten()

    # -------------------------
    # Updates
    # -------------------------
    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)
            L.log('train/CR_LOSS', critic_loss, step)

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self.scaler.unscale_(self.critic_optimizer)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=20.0)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()

        if self.log_param_hist_imgs:
            self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
            actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)
            actor_Q = torch.min(actor_Q1, actor_Q2)
            actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train/A_LOSS', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self.scaler.unscale_(self.actor_optimizer)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=20.0)
        self.scaler.step(self.actor_optimizer)

        if self.log_param_hist_imgs:
            self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)

        self.scaler.scale(alpha_loss).backward()
        self.scaler.step(self.log_alpha_optimizer)
        self.scaler.update()

    def update_cpc(self, obs_anchor, obs_pos, cpc_kwargs, L, step):
        _assert_ok_tensor(obs_anchor, "obs_anchor")
        _assert_ok_tensor(obs_pos, "obs_pos")

        if obs_anchor.device != obs_pos.device:
            raise RuntimeError(
                f"[CURL_SAC][BAD] obs_anchor.device != obs_pos.device: "
                f"{obs_anchor.device} vs {obs_pos.device}"
            )

        try:
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                z_a = self.CURL.encode(obs_anchor)
                z_pos = self.CURL.encode(obs_pos, ema=True)

            _assert_ok_tensor(z_a, "z_a")
            _assert_ok_tensor(z_pos, "z_pos")

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = self.CURL.compute_logits(z_a, z_pos)

            _assert_ok_tensor(logits, "logits")

            if logits.dim() != 2 or logits.shape[0] != logits.shape[1]:
                raise RuntimeError(
                    "[CURL_SAC][BAD] logits shape expected [B,B] but got "
                    f"{tuple(logits.shape)} | " + _tensor_brief(logits, "logits")
                )

            B = int(logits.shape[0])
            labels = torch.arange(B, device=logits.device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                loss = self.cross_entropy_loss(logits, labels)

            self.encoder_optimizer.zero_grad(set_to_none=True)
            self.cpc_optimizer.zero_grad(set_to_none=True)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.encoder_optimizer)
            self.scaler.unscale_(self.cpc_optimizer)

            torch.nn.utils.clip_grad_norm_(self.critic.encoder.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(self.CURL.parameters(), max_norm=10.0)

            self.scaler.step(self.encoder_optimizer)
            self.scaler.step(self.cpc_optimizer)
            self.scaler.update()

            if step % self.log_interval == 0:
                L.log('train/curl_loss', loss, step)
                L.log('train/CU_LOSS', loss, step)

        except RuntimeError:
            _safe_cuda_sync()
            msg = (
                "\n[CURL_SAC][EXCEPTION] update_cpc failed.\n"
                f"step={step}\n"
                f"{_tensor_brief(obs_anchor, 'obs_anchor')}\n"
                f"{_tensor_brief(obs_pos, 'obs_pos')}\n"
            )
            try:
                msg += f"{_tensor_brief(z_a, 'z_a')}\n"
            except Exception:
                pass
            try:
                msg += f"{_tensor_brief(z_pos, 'z_pos')}\n"
            except Exception:
                pass
            try:
                msg += f"{_tensor_brief(logits, 'logits')}\n"
            except Exception:
                pass
            print(msg)
            raise

    def update_predictive_cpc(self, obs, action, next_obs, L, step):
        _assert_ok_tensor(obs, "obs")
        _assert_ok_tensor(action, "action")
        _assert_ok_tensor(next_obs, "next_obs")

        z = self.critic.encoder(obs)
        z_next_pred = self.transition_model(z, action)
        z_next_target = self.critic_target.encoder(next_obs).detach()

        logits = torch.matmul(torch.matmul(z_next_pred, self.W_pred), z_next_target.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0]

        labels = torch.arange(logits.shape[0], device=logits.device, dtype=torch.long)
        loss = self.cross_entropy_loss(logits, labels)

        self.transition_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.encoder.parameters(), max_norm=20.0)
        torch.nn.utils.clip_grad_norm_(self.transition_model.parameters(), max_norm=20.0)
        self.transition_optimizer.step()

        if step % self.log_interval == 0:
            L.log('train/predictive_cpc_loss', loss, step)
            L.log('train/cpc_loss', loss, step)
            L.log('train/CPC_LOSS', loss, step)

    def update(self, replay_buffer, L, step, only_cpc=False):
        obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()

        _assert_ok_tensor(obs, "obs")
        _assert_ok_tensor(action, "action")
        _assert_ok_tensor(reward, "reward")
        _assert_ok_tensor(next_obs, "next_obs")
        _assert_ok_tensor(not_done, "not_done")

        # ✅ move to device (safe even if already on device)
        obs = obs.to(self.device, non_blocking=True)
        action = action.to(self.device, non_blocking=True)
        reward = reward.to(self.device, non_blocking=True)
        next_obs = next_obs.to(self.device, non_blocking=True)
        not_done = not_done.to(self.device, non_blocking=True)

        if reward.dtype != torch.float32:
            reward = reward.float()
        if not_done.dtype != torch.float32:
            not_done = not_done.float()

        if step % self.log_interval == 0:
            try:
                L.log('train/batch_reward', reward.mean(), step)
                L.log('train/BR', reward.mean(), step)
            except Exception:
                pass

        if not only_cpc:
            self.update_critic(obs, action, reward, next_obs, not_done, L, step)

            if step % self.actor_update_freq == 0:
                self.update_actor_and_alpha(obs, L, step)

            if step % self.critic_target_update_freq == 0:
                utils.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.critic_tau)
                utils.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.critic_tau)
                utils.soft_update_params(self.critic.encoder, self.critic_target.encoder, self.encoder_tau)

        if not self.pixel_sac:
            if step % self.cpc_update_freq == 0:
                obs_anchor, obs_pos = cpc_kwargs["obs_anchor"], cpc_kwargs["obs_pos"]
                obs_anchor = obs_anchor.to(self.device, non_blocking=True)
                obs_pos = obs_pos.to(self.device, non_blocking=True)
                self.update_cpc(obs_anchor, obs_pos, cpc_kwargs, L, step)
                if self.predictive_cpc:
                    self.update_predictive_cpc(obs, action, next_obs, L, step)

    # -------------------------
    # Save / load
    # -------------------------
    def save(self, model_dir, augmentation, step):
        torch.save(self.CURL.state_dict(), '%s/%s_curl_%s.pt' % (model_dir, augmentation, step))
        torch.save(self.actor.state_dict(), '%s/%s_actor_%s.pt' % (model_dir, augmentation, step))
        torch.save(self.critic.state_dict(), '%s/%s_critic_%s.pt' % (model_dir, augmentation, step))
        torch.save(
            {
                'transition_model': self.transition_model.state_dict(),
                'W_pred': self.W_pred.detach().cpu(),
                'predictive_cpc': self.predictive_cpc,
            },
            '%s/%s_predictive_cpc_%s.pt' % (model_dir, augmentation, step)
        )

    def load(self, model_dir, augmentation, step):
        self.CURL.load_state_dict(torch.load('%s/%s_curl_%s.pt' % (model_dir, augmentation, step), map_location=self.device))
        print('Loaded model %s/%s_curl_%s.pt' % (model_dir, augmentation, step))
        self.actor.load_state_dict(torch.load('%s/%s_actor_%s.pt' % (model_dir, augmentation, step), map_location=self.device))
        print('Loaded model %s/%s_actor_%s.pt' % (model_dir, augmentation, step))
        self.critic.load_state_dict(torch.load('%s/%s_critic_%s.pt' % (model_dir, augmentation, step), map_location=self.device))
        self.critic_target.load_state_dict(self.critic.state_dict())
        print('Loaded model %s/%s_critic_%s.pt' % (model_dir, augmentation, step))
        predictive_path = '%s/%s_predictive_cpc_%s.pt' % (model_dir, augmentation, step)
        if os.path.exists(predictive_path):
            predictive_state = torch.load(predictive_path, map_location=self.device)
            self.transition_model.load_state_dict(predictive_state['transition_model'])
            self.W_pred.data.copy_(predictive_state['W_pred'].to(self.device))
            self.predictive_cpc = bool(predictive_state.get('predictive_cpc', self.predictive_cpc))
            print('Loaded model %s/%s_predictive_cpc_%s.pt' % (model_dir, augmentation, step))


# -------------------------
# Optional baseline: TD3
# (cleaned so it won't break import)
# -------------------------
class TD3Agent(object):
    """TD3 baseline agent (minimal, independent)."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        augmentor,
        hidden_dim=1024,
        discount=0.99,
        tau=0.01,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=3e-4,
        critic_lr=3e-4,
    ):
        self.device = device
        self.augmentor = augmentor
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0
        self.image_shape = obs_shape[-2:]

        # NOTE: Actor here is stochastic Gaussian actor; we use mu as deterministic policy.
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim=50, log_std_min=-10, log_std_max=2,
            num_layers=4, num_filters=32
        ).to(device)

        self.actor_target = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim=50, log_std_min=-10, log_std_max=2,
            num_layers=4, num_filters=32
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim=50, num_layers=4, num_filters=32
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim=50, num_layers=4, num_filters=32
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.actor_target.train(training)
        self.critic_target.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().numpy().flatten()

    def sample_action(self, obs):
        a = self.select_action(obs)
        a = a + np.random.normal(0, 0.1, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def update(self, replay_buffer, L, step, only_cpc=False):
        if only_cpc:
            return

        obs, action, reward, next_obs, not_done, _ = replay_buffer.sample_cpc()

        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        not_done = not_done.to(self.device)

        with torch.no_grad():
            # deterministic target action: use mu
            next_mu, _, _, _ = self.actor_target(next_obs, compute_pi=False, compute_log_pi=False)
            noise = (torch.randn_like(next_mu) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (next_mu + noise).clamp(-1.0, 1.0)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            actor_loss = -self.critic.Q1(self.critic.encoder(obs), mu).mean()

            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

        if step % 1000 == 0:
            L.log('train_critic/loss_td3', critic_loss, step)

    def save(self, model_dir, augmentation, step):
        torch.save(self.actor.state_dict(), '%s/%s_td3_actor_%s.pt' % (model_dir, augmentation, step))
        torch.save(self.critic.state_dict(), '%s/%s_td3_critic_%s.pt' % (model_dir, augmentation, step))

    def load(self, model_dir, augmentation, step):
        self.actor.load_state_dict(torch.load('%s/%s_td3_actor_%s.pt' % (model_dir, augmentation, step)))
        self.critic.load_state_dict(torch.load('%s/%s_td3_critic_%s.pt' % (model_dir, augmentation, step)))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())


class DDPGAgent(object):
    """Pure DDPG Agent with extensibility for CURL/CPC."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        augmentor,
        hidden_dim=1024,
        discount=0.99,
        tau=0.01,
        actor_lr=3e-4,
        critic_lr=3e-4,
        # Following arguments are kept for future extensibility (CURL/CPC/etc.)
        encoder_feature_dim=50,
        num_layers=4,
        num_filters=32,
        pixel_sac=False,
        predictive_cpc=False
    ):
        self.device = device
        self.augmentor = augmentor
        self.discount = discount
        self.tau = tau
        self.image_shape = obs_shape[-2:]
        self.pixel_sac = pixel_sac
        self.predictive_cpc = predictive_cpc

        # DDPG typically uses a deterministic actor.
        # Our Actor class supports both; we'll use mu.
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, log_std_min=-10, log_std_max=2,
            num_layers=num_layers, num_filters=num_filters
        ).to(device)

        self.actor_target = Actor(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, log_std_min=-10, log_std_max=2,
            num_layers=num_layers, num_filters=num_filters
        ).to(device)

        # Pure DDPG uses only ONE critic (though our Critic class has two, we only use Q1)
        self.critic = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.actor_target.train(training)
        self.critic_target.train(training)

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            return mu.cpu().numpy().flatten()

    def sample_action(self, obs):
        # DDPG uses additive noise for exploration
        a = self.select_action(obs)
        a = a + np.random.normal(0, 0.1, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def update(self, replay_buffer, L, step, only_cpc=False):
        if only_cpc:
            return

        # sample_cpc() provides the batch and handles data augmentation
        obs, action, reward, next_obs, not_done, cpc_kwargs = replay_buffer.sample_cpc()

        obs = obs.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_obs = next_obs.to(self.device)
        not_done = not_done.to(self.device)

        # 1. Update Critic
        with torch.no_grad():
            # next_action from target actor (deterministic)
            next_mu, _, _, _ = self.actor_target(next_obs, compute_pi=False, compute_log_pi=False)
            # Pure DDPG: no noise on next_action, just target Q
            target_Q1, target_Q2 = self.critic_target(next_obs, next_mu)
            # Use only Q1 for pure DDPG
            target_Q = reward + (not_done * self.discount * target_Q1)

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 2. Update Actor (no delay in pure DDPG)
        mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
        actor_loss = -self.critic.Q1(self.critic.encoder(obs), mu).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # 3. Soft update targets
        utils.soft_update_params(self.critic, self.critic_target, self.tau)
        utils.soft_update_params(self.actor, self.actor_target, self.tau)

        if step % 1000 == 0:
            L.log('train_critic/loss_ddpg', critic_loss, step)
            L.log('train_actor/loss_ddpg', actor_loss, step)

    def save(self, model_dir, augmentation, step):
        torch.save(self.actor.state_dict(), '%s/%s_ddpg_actor_%s.pt' % (model_dir, augmentation, step))
        torch.save(self.critic.state_dict(), '%s/%s_ddpg_critic_%s.pt' % (model_dir, augmentation, step))

    def load(self, model_dir, augmentation, step):
        self.actor.load_state_dict(torch.load('%s/%s_ddpg_actor_%s.pt' % (model_dir, augmentation, step)))
        self.critic.load_state_dict(torch.load('%s/%s_ddpg_critic_%s.pt' % (model_dir, augmentation, step)))
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_target.load_state_dict(self.actor.state_dict())
