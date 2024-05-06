import torch
from multiprocessing import Pool
from AirfoilEnv import *
from tensormanager import TensorManager


class Train:
    def __init__(
        self,
        env,
        env_name,
        agent,
        epochs,
        mini_batch_size,
        n_iterations,
        num_points,
        horizon,
        number_of_trajectories,
        epsilon,
        beta=0.01,
    ):
        self.env = env
        self.env_name = env_name
        self.agent = agent
        self.epsilon = epsilon
        self.horizon = horizon
        self.epochs = epochs
        self.num_points = num_points
        self.number_of_trajectories = number_of_trajectories
        self.n_iterations = n_iterations
        self.mini_batch_size = mini_batch_size
        self.start_time = 0
        self.state = None  # Image Tensor
        self.running_reward = 0
        self.beta = beta
        self.steps_history = []
        self.rewards_history = []
        self.actor_loss_history = []
        self.critic_loss_history = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def step(self):
        for iteration in range(1, 1 + self.n_iterations):
            tensor_manager = TensorManager(
                env_num=self.number_of_trajectories,
                horizon=self.horizon,
                state_shape=(250, 500), # Image shape
                action_dim=self.agent.n_actions,
                device=self.device,
            )
            state = self.env.reset()
            state = state.to(self.device)

            with torch.no_grad(): 
                for n in range(self.number_of_trajectories):
                    # 1 episode (data collection)
                    for t in range(self.horizon):
                        # Actor
                        dist = self.agent.choose_dists(state, use_grad=False)
                        action = self.agent.choose_actions(dist)
                        scaled_actions = self.agent.scale_actions(action).numpy().squeeze()
                        log_prob = dist.log_prob(action).sum(dim=1)

                        # Critic
                        value = self.agent.get_value(state, use_grad=False)
                        next_state, reward = self.env.step(scaled_actions, t=t)

                        tensor_manager.update_tensors(
                            state,
                            action,
                            reward,
                            value,
                            log_prob,
                            n,
                            t,
                        )

                        state = torch.tensor(next_state, dtype=torch.float32).to(self.device)

                    next_value = self.agent.get_value(state, use_grad=False)
                    tensor_manager.values_tensor[:, -1] = next_value.squeeze()

            tensor_manager.advantages_tensor = self.get_gae(tensor_manager)
            # Train the agent
            actor_loss, critic_loss = self.train(tensor_manager)

            self.print_logs(iteration, actor_loss, critic_loss, t)

    def train(
        self,
        tensor_manager,
    ):
        total_actor_loss, total_critic_loss = 0, 0
        total_mini_batches = 0
        returns = (
            tensor_manager.advantages_tensor + tensor_manager.values_tensor[:, :-1]
        )
        for epoch in range(self.epochs):
            for (
                state,
                action,
                return_,
                adv,
                old_value,
                old_log_prob,
            ) in self.choose_mini_batch(
                self.mini_batch_size,
                tensor_manager.states_tensor,
                tensor_manager.actions_tensor,
                returns,
                tensor_manager.advantages_tensor,
                tensor_manager.values_tensor,
                tensor_manager.log_probs_tensor,
            ):
                state, action, return_, adv, old_value, old_log_prob = (
                    state.squeeze(),
                    action.squeeze(),
                    return_.squeeze(),
                    adv.squeeze(),
                    old_value.squeeze(),
                    old_log_prob.squeeze(),
                )
                state = state.unsqueeze(1)
                value = self.agent.get_value(state, use_grad=True)
                critic_loss = (return_ - value).pow(2).mean()

                new_dist = self.agent.choose_dists(state, use_grad=True)
                new_log_prob = new_dist.log_prob(action).sum(dim=1)
                ratio = (new_log_prob - old_log_prob).exp()

                actor_loss = self.compute_actor_loss(ratio, adv)

                entropy_loss = new_dist.entropy().mean()
                actor_loss -= self.beta * entropy_loss

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_mini_batches += 1

                self.agent.optimize(actor_loss, critic_loss)

            avg_actor_loss = total_actor_loss / total_mini_batches
            avg_critic_loss = total_critic_loss / total_mini_batches

        return avg_actor_loss, avg_critic_loss

    def get_gae(self, tensor_manager, gamma=0.99, lam=0.95):
        rewards = tensor_manager.rewards_tensor
        values = tensor_manager.values_tensor
        num_env, horizon = rewards.shape
        advs = torch.zeros_like(rewards).to(rewards.device)

        # Adjusting the values after the end of the episodes
        for env_idx in range(num_env):
            gae = 0
            for t in reversed(range(horizon)):
                delta = (
                    rewards[env_idx, t]
                    + gamma * values[env_idx, t + 1]
                    - values[env_idx, t]
                )
                gae = delta + gamma * lam * gae
                advs[env_idx, t] = gae
        return advs

    def compute_actor_loss(self, ratio, adv):
        pg_loss1 = adv * ratio
        pg_loss2 = adv * torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        loss = -torch.min(pg_loss1, pg_loss2).mean()

        return loss

    def choose_mini_batch(
        self,
        mini_batch_size,
        states,
        actions,
        returns,
        advs,
        values,
        log_probs,
    ):

        for _ in range(self.horizon // mini_batch_size):
            # 무작위로 mini_batch_size 개의 인덱스를 선택
            indices = torch.randperm(self.horizon)[:mini_batch_size].to(states.device)
            yield (
                states[:, indices],
                actions[:, indices],
                returns[:, indices],
                advs[:, indices],
                values[:, indices],
                log_probs[:, indices],
            )

    def print_logs(self, actor_loss, critic_loss, steps):

        running_reward = torch.mean(self.running_reward)
        # current_actor_lr = self.agent.actor_optimizer.param_groups[0]["lr"]
        # current_critic_lr = self.agent.critic_optimizer.param_groups[0]["lr"]

        self.steps_history.append(steps)
        self.rewards_history.append(running_reward.item())
        self.actor_loss_history.append(actor_loss)
        self.critic_loss_history.append(critic_loss)

        actor_loss = actor_loss.item() if torch.is_tensor(actor_loss) else actor_loss
        critic_loss = (
            critic_loss.item() if torch.is_tensor(critic_loss) else critic_loss
        )

        running_reward_val = torch.mean(self.running_reward).item()
        self.plot_and_save()

    def plot_and_save(self):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(self.steps_history, label="Average Steps")
        axs[0, 0].set_title("Average Steps")
        axs[0, 1].plot(self.rewards_history, label="Running Reward")
        axs[0, 1].set_title("Running Reward")
        axs[1, 0].plot(self.actor_loss_history, label="Actor Loss")
        axs[1, 0].set_title("Actor Loss")
        axs[1, 1].plot(self.critic_loss_history, label="Critic Loss")
        axs[1, 1].set_title("Critic Loss")

        for ax in axs.flat:
            ax.set_xlabel("Iteration")  # 모든 서브플롯에 x축 레이블 설정
            ax.set_ylabel("Value")  # 모든 서브플롯에 y축 레이블 설정
            ax.legend(loc="best")

        fig.subplots_adjust(hspace=0.2, wspace=0.2)

        plt.savefig(
            f"./results/results_graphs.png"
        )
        plt.close()
