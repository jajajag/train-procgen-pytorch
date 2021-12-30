from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels

import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name',         type=str, default = 'test',
            help='experiment name')
    parser.add_argument('--env_name',         type=str, default = 'starpilot',
            help='environment ID')
    parser.add_argument('--start_level',      type=int, default = int(0),
            help='start-level for environment')
    parser.add_argument('--num_levels',       type=int, default = int(500),
            help='number of training levels for environment')
    parser.add_argument('--distribution_mode',type=str, default = 'hard',
            help='distribution mode for environment')
    parser.add_argument('--param_name',       type=str, default = 'easy-200',
            help='hyper-parameter ID')
    parser.add_argument('--device',           type=str, default = 'gpu',
            required = False,
            help='whether to use gpu')
    parser.add_argument('--gpu_device',       type=int, default = int(0),
            required = False,
            help = 'visible device in CUDA')
    parser.add_argument('--num_timesteps',    type=int,
            # Easy for 25M timesteps, Hard for 200M timesteps
            default = int(200000000),
            help = 'number of training timesteps')
    parser.add_argument('--seed',             type=int,
            default = random.randint(0,9999),
            help='Random generator seed')
    parser.add_argument('--log_level',        type=int, default = int(40),
            help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints',  type=int, default = int(1),
            help='number of checkpoints to store')
    # JAG: Parameters for Crossnorm and Selfnorm
    parser.add_argument('--cnsn_type',        type=str, default = '',
            choices = ['', 'cn', 'sn', 'cnsn'],
            help='CNSN: if we use crossnorm and selfnorm')
    parser.add_argument('--pos',              type=str, default = 'post',
            choices = ['residual', 'identity', 'pre', 'post'],
            help='CNSN: position of the crossnorm and selfnorm layers')
    parser.add_argument('--beta',             type=float, default = float(1),
            help='CNSN: parameter of beta distribution')
    parser.add_argument('--crop',             type=str, default = 'neither',
            choices = ['neither', 'style', 'content', 'both'],
            help='CNSN: if we do crop to style or content')
    parser.add_argument('--eval_env',             type=bool, default = True,
            help='use evaluation environment or not')

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    num_timesteps = args.num_timesteps
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    # JAG: Parse CNSN parameters
    cnsn_type = args.cnsn_type
    pos = args.pos
    beta = args.pos
    crop = args.pos

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    ############
    ## DEVICE ##
    ############
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)
    device = torch.device('cuda')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    # By default, pytorch utilizes multi-threaded cpu
    # Procgen is able to handle thousand of steps on a single core
    torch.set_num_threads(1)
    env = ProcgenEnv(num_envs=n_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=distribution_mode)
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        # normalizing returns, but not the img frames.
        env = VecNormalize(env, ob=False)
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    # JAG: If we use eval_env
    eval_env = None
    if args.eval_env:
        eval_env = ProcgenEnv(
                num_envs=n_envs, env_name=env_name,
                # We do not limit num_levels for eval_env
                start_level=start_level, num_levels=0,
                distribution_mode=distribution_mode)
        normalize_rew = hyperparameters.get('normalize_rew', True)
        eval_env = VecExtractDictObs(eval_env, "rgb")
        if normalize_rew:
            # normalizing returns, but not the img frames.
            eval_env = VecNormalize(eval_env, ob=False)
        eval_env = TransposeFrame(eval_env)
        eval_env = ScaledFloatFrame(eval_env)

    ############
    ## LOGGER ##
    ############
    print('INITIALIZAING LOGGER...')
    logdir = 'procgen/' + env_name + '/' + exp_name + '/' + 'seed' + '_' + \
             str(seed) + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('logs', logdir)
    if not (os.path.exists(logdir)):
        os.makedirs(logdir)
    # JAG: Add eval env
    logger = Logger(n_envs, logdir, args.eval_env)

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels,
                pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type)

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(
            observation_shape, hidden_state_dim, n_steps, n_envs, device)

    ###########
    ## AGENT ##
    ###########
    print('INTIALIZING AGENT...')
    algo = hyperparameters.get('algo', 'ppo')
    if algo == 'ppo':
        from agents.ppo import PPO as AGENT
    else:
        raise NotImplementedError
    agent = AGENT(env, policy, logger, storage, device, num_checkpoints, 
            eval_env=eval_env, **hyperparameters)

    ##############
    ## TRAINING ##
    ##############
    print('START TRAINING...')
    agent.train(num_timesteps)
