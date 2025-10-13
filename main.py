import os
import numpy as np
import random
import jax
from datetime import datetime

from dataclasses import dataclass
from typing import Optional, Tuple, Literal
import tyro

from bayesrl.environments import LazyChain, GridWorld, Loop, Chain, DeepSeaGym
from bayesrl.agents import ThompsonSampAgent, EUBRLAgent, QLearningAgent, RMAXAgent, SARSAAgent, VBRBAgent, BEBAgent
from bayesrl.trial import Trial


@dataclass
class Args:
    # General
    agent_name: Literal['PSRL', 'EUBRL', 'QLearning', 'RMAX', 'SARSA', 'VBRB', 'BEB'] = 'EUBRL'
    """the agent you wish to choose"""
    store_dir: Optional[str] = None
    """directory to save data"""
    suffix: Optional[str] = None
    """directory to save data"""
    seed: Optional[int] = None
    """seed of the experiment"""
    num_trials: int = 1
    """number of trials"""
    num_environment_steps: int = 10000
    """number of total environment steps"""
    num_episodes_for_success: int = 10
    """number of episodes counted as success for an algorithm"""
    policy_update_interval: int = 10
    """frequency of policy update"""
    figure_name: Optional[str] = None
    """name of the figure to save"""
    use_eubrl_reward: bool = True
    """whether to use the EUBRL reward for epistemically guided exploration"""
    use_transition_sampling: bool = True
    """whether to sample from the belief for PSRL"""
    use_normal_gamma_prior: bool = False
    """whether to use Normal-Gamma prior for reward"""
    use_alertness: bool = False
    """whether to use the alertness of instant reward modeling when using Normal-Normal"""
    use_sqrt_last: bool = False
    """whether to take square root after merging epistemic uncertainties from reward and transition"""
    use_independent_max: bool = False
    """whether to track maximum epistemic uncertainty for each state-action"""
    eu_scale: float = 1.0
    """the scaling factor of epistemic uncertainty"""
    reward_eu_scale: float = 1.0
    """the scaling factor of epistemic uncertainty in reward"""
    transition_eu_scale: float = 1.0
    """the scaling factor of epistemic uncertainty in transition"""

    # Environment
    env_name: Literal['LazyChain', 'GridWorld', 'Bipolar', 'Loop', 'Chain', 'DeepSea'] = 'LazyChain'

    ## LazyChain
    chain_size: int = 4
    p_error: float = 0.0
    discount_factor: float = 0.99
    dirichlet_param: Optional[float] = None

    ## GridWorld
    grid_layout: Literal['trivial', 'larger', 'bipolar'] = 'larger'
    action_error_prob: float = 0.1
    optimal_goal_reward: float = 5
    suboptimal_goal_reward: float = 5
    suboptimal_goal_reward_std: float = 2
    move_reward: float = -1
    hit_reward: float = -1

    ## Loop
    loop_length: int = 5
    num_loops: int = 2

    ## DeepSea
    deepsea_stochastic: bool = False
    deepsea_size: int = 10
    deepsea_randomize_actions: bool = True

    ## Chain
    chain_left_reward: float = 2
    chain_right_reward: float = 10

    # Algorithm
    use_jax: bool = True
    """whether to use jax for value iteration"""
    eu_type: Literal['One-hot', 'Product', 'Information Gain'] = 'One-hot'
    """what type of epistemic uncertainty to choose"""
    reward_param: Optional[float] = None
    """initial value of the reward estimate when not using priors"""
    known_reward: bool = False
    """whether ground truth reward is provided"""
    instant_reward: bool = True
    """whether to model reward as r(s, a, s')"""
    alertness_scale: float = 100
    """scale of the actual variance for alertness when used with Normal-Normal"""
    alertness_max_eu: float = 1000
    """initial maximum epistemic uncertainty when using alertness"""

    ## R-max
    min_visit_count: int = 10
    """The minimum number of visitation counts for (s, a) to be known"""

    # Belief
    ## Normal-Normal
    tau: Optional[float] = None
    """precision of the Normal prior"""
    precision: Optional[float] = None
    """known precision of the source distribution"""

    ## Normal-Gamma
    beta: float = 1.0
    """single consolidated parameter for Normal-Gamma"""


def make_env(args):
    env_name = args.env_name
    if env_name == "LazyChain":
        env = LazyChain(
            left_length=args.chain_size,
            left_reward=args.chain_size - 1,
            right_length=args.chain_size,
            right_reward=2 * args.chain_size - 1,
            on_chain_reward=-1,
            p_error=args.p_error,
            random_state=args.seed,
        ) # Deterministic if p_error = 0, otherwise stochastic with probability p to flip the outcome
    elif env_name == "GridWorld":
        env = GridWorld(
            GridWorld.samples[args.grid_layout],
            action_error_prob=args.action_error_prob,
            rewards={'*': args.optimal_goal_reward, 'moved': args.move_reward, 'hit-wall': args.hit_reward},
        )
    elif env_name == "Bipolar":
        env = GridWorld(
            GridWorld.samples['bipolar'],
            terminal_markers="*$",
            rewards={'*': args.optimal_goal_reward, '$': (args.suboptimal_goal_reward, args.suboptimal_goal_reward_std), 'moved': args.move_reward, 'hit-wall': args.hit_reward},
        )
    elif env_name == "Loop":
        env = Loop(
            loop_length=args.loop_length,
            num_loops=args.num_loops,
        )
    elif env_name == "Chain":
        env = Chain(
            right_length=args.chain_size,
            left_reward=args.chain_left_reward,
            right_reward=args.chain_right_reward,
        )
    elif env_name == "DeepSea":
        env = DeepSeaGym(
            size=args.deepsea_size,
            deterministic=(not args.deepsea_stochastic),
            randomize_actions=args.deepsea_randomize_actions,
            seed=args.seed,     # for reproducibility
        )
    else:
        raise NotImplementedError

    return env


def make_agent(args, env):
    # Get maximum reward
    max_reward = env.get_max_reward()

    if args.use_jax:
        rng_key = jax.random.PRNGKey(args.seed)
    else:
        rng_key = None

    if args.known_reward:
        env_reward = env.get_reward(use_instant_reward=args.instant_reward)
    else:
        env_reward = None

    if args.tau is None:
        if args.env_name == "LazyChain":
            args.tau = 1 / args.chain_size
        else:
            args.tau = 1 / 100   # set default one if not given

    if args.precision is None:
        args.precision = args.tau * 100

    if args.dirichlet_param is None:
        args.dirichlet_param = 1 / env.num_states # 1 / S if Dirichlet parameter is not given


    if args.agent_name == "EUBRL":
        agent = EUBRLAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            discount_factor=args.discount_factor,
            T=args.policy_update_interval,
            max_reward=max_reward,
            env_name=args.env_name,
            dirichlet_param=args.dirichlet_param,
            reward_param=args.reward_param,
            env_reward=env_reward,
            tau=args.tau,
            precision=args.precision,
            beta=args.beta,
            use_jax=args.use_jax,
            eu_type=args.eu_type,
            rng_key=rng_key,
            eu_scale=args.eu_scale,
            transition_eu_scale=args.transition_eu_scale,
            reward_eu_scale=args.reward_eu_scale,
            use_eubrl_reward=args.use_eubrl_reward,
            use_normal_gamma_prior=args.use_normal_gamma_prior,
            instant_reward=args.instant_reward,
            use_alertness=args.use_alertness,
            use_sqrt_last=args.use_sqrt_last,
            alertness_scale=args.alertness_scale,
            alertness_max_eu=args.alertness_max_eu,
            use_independent_max=args.use_independent_max,
        )
    elif args.agent_name == "VBRB":
        agent = VBRBAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            discount_factor=args.discount_factor,
            T=args.policy_update_interval,
            max_reward=max_reward,
            dirichlet_param=args.dirichlet_param,
            reward_param=args.reward_param,
            env_reward=env_reward,
            tau=args.tau,
            precision=args.precision,
            beta=args.beta,
            use_jax=args.use_jax,
            env_name=args.env_name,
            rng_key=rng_key,
            transition_var_scale=args.transition_eu_scale,
            reward_var_scale=args.reward_eu_scale,
            transition_var_scale=args.eu_scale,
            reward_var_scale=args.eu_scale,
            use_normal_gamma_prior=args.use_normal_gamma_prior,
        )
    elif args.agent_name == "BEB":
        agent = BEBAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            discount_factor=args.discount_factor,
            T=args.policy_update_interval,
            max_reward=max_reward,
            env_name=args.env_name,
            dirichlet_param=args.dirichlet_param,
            reward_param=args.reward_param,
            env_reward=env_reward,
            tau=args.tau,
            precision=args.precision,
            beta=args.beta,
            use_jax=args.use_jax,
            eu_type=args.eu_type,
            rng_key=rng_key,
            eu_scale=args.eu_scale,
            use_normal_gamma_prior=args.use_normal_gamma_prior,
            instant_reward=args.instant_reward,
            use_alertness=args.use_alertness,
            alertness_scale=args.alertness_scale,
            alertness_max_eu=args.alertness_max_eu,
        )
    elif args.agent_name == "RMAX":
        agent = RMAXAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            discount_factor=args.discount_factor,
            T=args.policy_update_interval,
            max_reward=max_reward if args.reward_param is None else args.reward_param,
            min_visit_count=args.min_visit_count,
            use_max_reward=True,
            use_jax=args.use_jax,
            rng_key=rng_key,
            env_name=args.env_name,
        )
    elif args.agent_name == "PSRL":
        agent = ThompsonSampAgent(
            num_states=env.num_states,
            num_actions=env.num_actions,
            discount_factor=args.discount_factor,
            T=args.policy_update_interval,
            max_reward=max_reward,
            dirichlet_param=args.dirichlet_param,
            reward_param=args.reward_param,
            env_reward=env_reward,
            tau=args.tau,
            precision=args.precision,
            beta=args.beta,
            use_jax=args.use_jax,
            rng_key=rng_key,
            use_normal_gamma_prior=args.use_normal_gamma_prior,
            transition_sampling=args.use_transition_sampling,
            env_name=args.env_name,
            instant_reward=args.instant_reward,
        )
    elif args.agent_name == "QLearning":
        agent = QLearningAgent(
           num_states=env.num_states,
           num_actions=env.num_actions,
           discount_factor=args.discount_factor,
           learning_rate=0.01,
           epsilon=0.1,
        )
    elif args.agent_name == "SARSA":
        agent = SARSAAgent(
           num_states=env.num_states,
           num_actions=env.num_actions,
           discount_factor=args.discount_factor,
           learning_rate=0.01,
           epsilon=0.1
        )
    else:
        raise NotImplementedError

    trial = Trial(
        agent,
        env,
        env_name=args.env_name,
        min_iterations=args.num_environment_steps,
        num_episodes_for_success=args.num_episodes_for_success
    )
    trial.run_multiple(args.num_trials)

    return agent, trial


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # set seed for reproducibility
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)  # Random seed in [0, 2**32 - 1] if the seed is not given.
        print(f"Random seed {args.seed} generated!")

    # For reproducibility
    set_seed(args.seed)

    # env
    env = make_env(args)

    # agent
    agent, trial = make_agent(args, env)

    # Save data
    if args.store_dir is not None:

        if args.env_name == "LazyChain":
            env_type = "stochastic" if args.p_error > 0 else "deterministic"
            path = f"{args.store_dir}/{args.env_name}/{env_type}/{args.chain_size}/{args.agent_name}"
        elif args.env_name == "DeepSea":
            env_type = "stochastic" if args.deepsea_stochastic else "deterministic"
            path = f"{args.store_dir}/{args.env_name}/{env_type}/{args.deepsea_size}/{args.agent_name}"
        elif args.env_name == "Loop":
            path = f"{args.store_dir}/{args.env_name}/{args.num_loops}/{args.agent_name}"
        elif args.env_name == "Chain":
            path = f"{args.store_dir}/{args.env_name}/{args.agent_name}"
        else:
            raise NotImplementedError

        if args.suffix is not None and isinstance(args.suffix, str):
            path = path + "-" + args.suffix

        filename = os.path.join(path, str(args.seed))

        os.makedirs(path, exist_ok=True)

        data_to_store = {
                "rewards_by_episode": trial.array_rewards_by_episode,
                "iteration_by_episode": trial.array_iteration_by_episode,
                "success_by_episode": trial.array_success_by_episode,
                "rewards_by_iteration": trial.array_rewards_by_iteration,
                "metrics": trial.array_metrics,
        }

        print(" Storing data ...")
        np.savez(filename, **data_to_store)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
