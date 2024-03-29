import torch

class Train:
    def __init__(
        self,
        env,
        env_name,
        n_iterations,
        agent,
        epochs,
        mini_batch_size,
        epsilon,
        horizon,
    ):
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.n_iterations = n_iterations
        self.env_num = env.env_batch
        self.start_time = 0
        self.running_reward = 0
        self.steps_history = []
        self.rewards_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []

    def choose_mini_batch(
        self,
        mini_batch_size,
        states,
        actions,
        action_maps,
        returns,
        advs,
        values,
        log_probs,
        time_step_tensor,
    ):
        full_batch_size = states.size(0)
        for _ in range(full_batch_size // mini_batch_size):
            # 무작위로 mini_batch_size 개의 인덱스를 선택
            indices = torch.randperm(full_batch_size)[:mini_batch_size].to(
                states.device
            )
            yield (
                states[indices],
                actions[indices],
                action_maps[indices],
                returns[indices],
                advs[indices],
                values[indices],
                log_probs[indices],
                time_step_tensor[indices],
            )

    def train(
        self,
        tensor_manager,
    ):
        returns = tensor_manager.advantages_tensor + tensor_manager.values_tensor

        for epoch in range(self.epochs):
            for (
                state,
                action,
                action_map,
                return_,
                adv,
                old_value,
                old_log_prob,
                time_step,
            ) in self.choose_mini_batch(
                self.mini_batch_size,
                tensor_manager.states_tensor,
                tensor_manager.actions_tensor,
                tensor_manager.action_maps_tensor,
                returns,
                tensor_manager.advantages_tensor,
                tensor_manager.values_tensor,
                tensor_manager.log_probs_tensor,
                tensor_manager.time_step_tensor,
            ):

                # 업데이트된 숨겨진 상태를 사용하여 critic 및 actor 업데이트
                value = self.agent.get_value(state, action_map, use_grad=True)

                critic_loss = (return_ - value).pow(2).mean()

                new_dist = self.agent.choose_dists(state, action_map, use_grad=True)
                new_log_prob = new_dist.log_prob(action).sum(dim=1)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_actor_loss(ratio, adv)

                entropy_loss = new_dist.entropy().mean()

                actor_loss += -0.03 * entropy_loss

                self.agent.optimize(actor_loss, critic_loss)

        return actor_loss, critic_loss
    

    def compute_returns_and_advantages(rewards, values, next_value, gamma=0.99, lam=0.95):
        """
        GAE(Generalized Advantage Estimation)로 반환값과 이점을 계산합니다.

        Args:
        - rewards (list of float): 각 타임스텝에서 얻은 보상의 리스트.
        - values (list of float): 에이전트의 가치 함수로부터 얻은 각 타임스텝의 가치 예측값의 리스트.
        - next_value (float): 에피소드 종료 후의 다음 상태의 가치 예측값.
        - gamma (float): 할인 계수.
        - lam (float): GAE 람다 파라미터.

        Returns:
        - returns (torch.Tensor): 각 타임스텝에 대한 반환값.
        - advantages (torch.Tensor): 각 타임스텝에 대한 이점.
        """
        gae = 0
        returns = []
        advantages = []
        # 마지막 타임스텝에서의 다음 가치를 초기값으로 설정
        next_return = next_value

        # 각 타임스텝에 대해 역순으로 계산
        for step in reversed(range(len(rewards))):
            # 해당 타임스텝에서의 반환값 계산
            delta = rewards[step] + gamma * next_return - values[step]
            gae = delta + gamma * lam * gae
            next_return = values[step] + gae
            returns.insert(0, next_return)
            advantages.insert(0, gae)

        # 리스트를 텐서로 변환
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)
        # 이점을 정규화하여 학습의 안정성을 개선할 수 있음
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        return returns, advantages

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()

        return loss
