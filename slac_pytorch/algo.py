import os

import numpy as np
import torch as th
from torch.optim import Adam

from slac_pytorch.buffer import ReplayBuffer
from slac_pytorch.models.slac import GaussianPolicy, LatentModel, TwinnedQNetwork
from slac_pytorch.models.nctrl import NCTRL, NCTRL, CTDRL, NCTRLzLatent
from slac_pytorch.utils import create_feature_actions, grad_false, soft_update


class SlacAlgorithm:
    """
    Stochactic Latent Actor-Critic(SLAC).

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        action_repeat,
        device,
        args,
        discrete=False
    ):
        assert args is not None, "No configuration settings"
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)

        # Replay buffer.
        self.buffer = ReplayBuffer(args.buffer_size, args.num_sequences, state_shape, action_shape, device)

        # Networks.
        self.actor = GaussianPolicy(action_shape, args.num_sequences, args.feature_dim, args.hidden_units).to(device)
        
        if len(args.actor_path) > 0:
            self.actor.load_state_dict(th.load(args.actor_path))
            
        self.critic = TwinnedQNetwork(action_shape, args.z1_dim, args.z2_dim, args.hidden_units).to(device)
    
        if len(args.critic_path) > 0:
            self.critic.load_state_dict(th.load(args.critic_path))
            
        self.critic_target = TwinnedQNetwork(action_shape, args.z1_dim, args.z2_dim, args.hidden_units).to(device)
    
        if len(args.critic_path) > 0:
            self.critic_target.load_state_dict(th.load(args.critic_path))
            
        self.latent = LatentModel(state_shape, action_shape, args.feature_dim, args.z1_dim, args.z2_dim, args.hidden_units).to(device)
        
        if len(args.latent_path) > 0:
            self.latent.load_state_dict(th.load(args.latent_path))
            
        soft_update(self.critic_target, self.critic, 1.0)
        grad_false(self.critic_target)

        # Target entropy is -|A|.
        self.target_entropy = -float(action_shape[0])
        # We optimize log(alpha) because alpha is always bigger than 0.
        self.log_alpha = th.zeros(1, requires_grad=True, device=device)
        with th.no_grad():
            self.alpha = self.log_alpha.exp()

        # Optimizers.
        self.optim_actor = Adam(self.actor.parameters(), lr=args.lr_sac)
        self.optim_critic = Adam(self.critic.parameters(), lr=args.lr_sac)
        self.optim_alpha = Adam([self.log_alpha], lr=args.lr_sac)
        self.optim_latent = Adam(self.latent.parameters(), lr=args.lr_latent)

        self.learning_steps_sac = 0
        self.learning_steps_latent = 0
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.action_repeat = action_repeat
        self.device = device
        self.gamma = args.gamma
        self.batch_size_sac = args.batch_size_sac
        self.batch_size_latent = args.batch_size_latent
        self.num_sequences = args.num_sequences
        self.tau = args.tau
        self.discrete = discrete
        # JIT compile to speed up.
        fake_feature = th.empty(1, args.num_sequences + 1, args.feature_dim, device=device)
        fake_action = th.empty(1, args.num_sequences, action_shape[0], device=device)
        self.create_feature_actions = th.jit.trace(create_feature_actions, (fake_feature, fake_action))

    def preprocess(self, ob):
        state = th.tensor(ob.state, dtype=th.uint8, device=self.device).float().div_(255.0)
        with th.no_grad():
            feature = self.latent.encoder(state).view(1, -1)
        action = th.tensor(ob.action, dtype=th.float, device=self.device)
        feature_action = th.cat([feature, action], dim=1)
        return feature_action

    def explore(self, ob):
        feature_action = self.preprocess(ob)
        with th.no_grad():
            action = self.actor.sample(feature_action)[0].cpu().numpy()[0]

        if self.discrete:
            action = int(action[0] > 0.5)

        return action

    def exploit(self, ob):
        feature_action = self.preprocess(ob)
        with th.no_grad():
            action = self.actor(feature_action).cpu().numpy()[0]

        if self.discrete:
            action = int(action[0] > 0.5)

        return action

    def step(self, env, ob, t, is_random):
        t += 1

        if is_random:
            action = env.action_space.sample()
        else:
            action = self.explore(ob)
                
        state, reward, done, truncated, infos = env.step(action)
        mask = False if t == env.spec.max_episode_steps else done
        ob.append(state, action)
        self.buffer.append(action, reward, mask, state, done)

        if done:
            t = 0
            state, _ = env.reset()
            ob.reset_episode(state)
            self.buffer.reset_episode(state)

        return t

    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)

        loss_kld, loss_image, loss_reward = self.latent.calculate_loss(state_, action_, reward_, done_)

        self.optim_latent.zero_grad()
        (loss_kld + loss_image + loss_reward).backward()
        self.optim_latent.step()

        if self.learning_steps_latent % 1000 == 0:
            writer.add_scalar("loss/kld", loss_kld.item(), self.learning_steps_latent)
            writer.add_scalar("loss/reward", loss_reward.item(), self.learning_steps_latent)
            writer.add_scalar("loss/image", loss_image.item(), self.learning_steps_latent)

    def update_sac(self, writer):
        self.learning_steps_sac += 1
        state_, action_, reward, done = self.buffer.sample_sac(self.batch_size_sac)
        z, next_z, action, feature_action, next_feature_action = self.prepare_batch(state_, action_)
        self.update_critic(z, next_z, action, next_feature_action, reward, done, writer)
        self.update_actor(z, feature_action, writer)
        soft_update(self.critic_target, self.critic, self.tau)

    def prepare_batch(self, state_, action_):
        with th.no_grad():
            # f(1:t+1)
            feature_ = self.latent.encoder(state_)
            # z(1:t+1)
            z_ = th.cat(self.latent.sample_posterior(feature_, action_)[2:4], dim=-1)

        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def update_critic(self, z, next_z, action, next_feature_action, reward, done, writer):
        curr_q1, curr_q2 = self.critic(z, action)
        with th.no_grad():
            next_action, log_pi = self.actor.sample(next_feature_action)
            next_q1, next_q2 = self.critic_target(next_z, next_action)
            next_q = th.min(next_q1, next_q2) - self.alpha * log_pi
        target_q = reward + (1.0 - done) * self.gamma * next_q
        loss_critic = (curr_q1 - target_q).pow_(2).mean() + (curr_q2 - target_q).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        self.optim_critic.step()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/critic", loss_critic.item(), self.learning_steps_sac)

    def update_actor(self, z, feature_action, writer):
        action, log_pi = self.actor.sample(feature_action)
        q1, q2 = self.critic(z, action)
        loss_actor = -th.mean(th.min(q1, q2) - self.alpha * log_pi)

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.optim_actor.step()

        with th.no_grad():
            entropy = -log_pi.detach().mean()
        loss_alpha = -self.log_alpha * (self.target_entropy - entropy)

        self.optim_alpha.zero_grad()
        loss_alpha.backward(retain_graph=False)
        self.optim_alpha.step()
        with th.no_grad():
            self.alpha = self.log_alpha.exp()

        if self.learning_steps_sac % 1000 == 0:
            writer.add_scalar("loss/actor", loss_actor.item(), self.learning_steps_sac)
            writer.add_scalar("loss/alpha", loss_alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/alpha", self.alpha.item(), self.learning_steps_sac)
            writer.add_scalar("stats/entropy", entropy.item(), self.learning_steps_sac)

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        th.save(self.latent.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        th.save(self.latent.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
        th.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        th.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        th.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))


class NCTRLAlgorithm(SlacAlgorithm):
    """
    Stochactic Latent Actor-Critic(SLAC) with NCTRL framework for latent model. 

    Paper: https://arxiv.org/abs/1907.00953
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        action_repeat,
        device,
        args
    ):
        assert args is not None, "No configuration settings"
        np.random.seed(args.seed)
        th.manual_seed(args.seed)
        th.cuda.manual_seed(args.seed)
        super().__init__(state_shape=state_shape, action_shape=action_shape, action_repeat=action_repeat, device=device, args=args)
            
        self.latent = NCTRLzLatent(x_dim=64, z_dim=32, lags=2, n_class=4, hidden_dim=256, embedding_dim=8, lr=5e-4, beta=2.0e-3, gamma=2.0e-3)
        
        if len(args.latent_path) > 0:
            self.latent.load_state_dict(th.load(args.latent_path))
    
        # Optimizers.
        self.optim_latent = Adam(self.latent.parameters(), lr=args.lr_latent)
        self.model_opt = th.optim.AdamW(self.latent.net.parameters(), lr=5.0e-4, betas=(0.9, 0.999), weight_decay=0.0001)
        self.hmm_opt = th.optim.Adam(self.latent.hmm.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0001)

    def preprocess(self, ob):
        state = th.tensor(ob.state, dtype=th.uint8, device=self.device).float().div_(255.0)
        with th.no_grad():
            feature, mu, logvar = self.latent.net.encode(state)
        
        action = th.tensor(ob.action, dtype=th.float, device=self.device)
        feature_action = th.cat([feature.view(1, -1), action], dim=1)
        return feature_action


    def update_latent(self, writer):
        self.learning_steps_latent += 1
        state_, action_, reward_, done_ = self.buffer.sample_latent(self.batch_size_latent)

        loss, recon_loss, hmm_loss, kld_normal, kld_laplace = self.latent.calculate_loss(state_, action_, reward_, done_)

        self.hmm_opt.zero_grad()
        hmm_loss.backward()
        self.hmm_opt.step()

        self.model_opt.zero_grad()
        loss.backward()
        self.model_opt.step()

        if self.learning_steps_latent % 1000 == 0:
            writer.add_scalar("loss/kld_normal", kld_normal.item(), self.learning_steps_latent)
            writer.add_scalar("loss/kld_laplace", kld_laplace.item(), self.learning_steps_latent)
            writer.add_scalar("loss/kld_laplace", kld_laplace.item(), self.learning_steps_latent)
            writer.add_scalar("loss/hmm_loss", hmm_loss.item(), self.learning_steps_latent)
            writer.add_scalar("loss/recon_loss", recon_loss.item(), self.learning_steps_latent)
            writer.add_scalar("loss/elbo_loss", loss.item(), self.learning_steps_latent)

            # writer.add_scalar("loss/reward", loss_reward.item(), self.learning_steps_latent)
            # writer.add_scalar("loss/image", loss_image.item(), self.learning_steps_latent)

    def prepare_batch(self, state_, action_):

        with th.no_grad():
            # f(1:t+1)
            feature_, _ , _  = self.latent.net.encode(state_)
            # z(1:t+1)
            z_ = self.latent.sample_prior(feature_, action_)
        # z(t), z(t+1)
        z, next_z = z_[:, -2], z_[:, -1]
        # a(t)
        action = action_[:, -1]
        # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
        feature_action, next_feature_action = self.create_feature_actions(feature_, action_)

        return z, next_z, action, feature_action, next_feature_action

    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # We don't save target network to reduce workloads.
        th.save(self.latent.net.encoder.state_dict(), os.path.join(save_dir, "encoder.pth"))
        th.save(self.latent.net.decoder.state_dict(), os.path.join(save_dir, "decoder.pth"))
        th.save(self.latent.state_dict(), os.path.join(save_dir, "latent.pth"))
        th.save(self.actor.state_dict(), os.path.join(save_dir, "actor.pth"))
        th.save(self.critic.state_dict(), os.path.join(save_dir, "critic.pth"))
