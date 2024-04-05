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
        t,
    ):
        self.states_tensor[:, t] = states
        self.actions_tensor[:, t] = actions
        self.rewards_tensor[:, t] = rewards
        self.values_tensor[:, t] = values.squeeze()
        self.log_probs_tensor[:, t] = log_probs
