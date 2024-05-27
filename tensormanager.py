import torch


class TensorManager:
    def __init__(
        self,
        env_num,
        horizon,
        state_shape,
        action_dim,
        device,
    ):
        self.env_num = env_num
        self.horizon = horizon
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.device = device

        self.states_tensor = self.init_tensor(
            [self.env_num, self.horizon, *self.state_shape], False
        )
        self.actions_tensor = self.init_tensor(
            [self.env_num, self.horizon, self.action_dim], False
        )
        self.rewards_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.values_tensor = self.init_tensor([self.env_num, self.horizon + 1], False)
        self.log_probs_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.advantages_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.return_tensor = self.init_tensor([self.env_num, self.horizon], False)
        self.time_step_tensor = torch.arange(
            0, self.horizon, device=self.device
        ).repeat(self.env_num, 1)

    def init_tensor(self, shape, requires_grad):
        return torch.zeros(shape, requires_grad=requires_grad).to(self.device)

    def update_tensors(
        self,
        states,
        actions,
        rewards,
        values,
        log_probs,
        traj_idx,
        t,
    ):
        self.states_tensor[traj_idx, t, :] = states
        self.actions_tensor[traj_idx, t] = actions
        self.rewards_tensor[traj_idx, t] = rewards
        self.values_tensor[traj_idx, t] = values.squeeze()
        self.log_probs_tensor[traj_idx, t] = log_probs

    def flatten_tensors(self):
        self.states_tensor = self.states_tensor.view(
            self.env_num * self.horizon, *self.state_shape
        )
        self.actions_tensor = self.actions_tensor.view(
            self.env_num * self.horizon, self.action_dim
        )
        self.return_tensor = self.return_tensor.view(self.env_num * self.horizon)
        self.rewards_tensor = self.rewards_tensor.view(self.env_num * self.horizon)
        self.values_tensor = self.values_tensor.view(self.env_num * (self.horizon + 1))
        self.log_probs_tensor = self.log_probs_tensor.view(self.env_num * self.horizon)
        self.advantages_tensor = self.advantages_tensor.view(
            self.env_num * self.horizon
        )
