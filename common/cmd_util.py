import argparse


class ArgumentParser:
    """
    Custom Argument parser that differentiates between the different algorithms that are implemented in this repository
    and adds custom arguments for each algorithm.
    """

    @staticmethod
    def parse():
        parser = argparse.ArgumentParser(description='Solve the different gym environments')

        # ---------------------
        #  General Arguments
        # ---------------------

        # parser.add_argument('experiment_id', type=str, help='identifier to store experiment results')
        parser.add_argument('--name', type=str, required=True, help='identifier to store experiment results')
        parser.add_argument('--env', type=str, required=True, help='name of the environment to be learned')
        parser.add_argument('--robot', action='store_true', help='run the experiment using the real robot environment')

        parser.add_argument('--n_epochs', type=int, default=50, help='number of training epochs')
        parser.add_argument('--n_steps', type=int, default=3000, help='number of environment steps per epoch')
        parser.add_argument('--seed', type=int, help='seed for torch/numpy/gym to make experiments reproducible')

        parser.add_argument('--render', action='store_true', help='render the environment')

        group = parser.add_mutually_exclusive_group()
        group.add_argument('--eval', action='store_true', help='toggles evaluation mode')
        group.add_argument('--resume', action='store_true',
                           help='resume training on an existing model by loading the last checkpoint')

        subparsers = parser.add_subparsers(description='Algorithms to choose from', dest='command')

        # ---------------------
        #  REPS Arguments
        # ---------------------
        reps_parser = subparsers.add_parser('REPS')
        reps_parser.add_argument('--epsilon', type=float, default=0.1, help='KL constraint.')
        reps_parser.add_argument('--gamma', type=float, default=0.99, help='1 minus environment reset probability.')

        reps_parser.add_argument('--n_fourier', type=int, default=75, help='number of fourier features.')
        reps_parser.add_argument('--fourier_band', type=float, nargs='+', help='number of fourier features.')

        # ---------------------
        #  ACREPS Arguments
        # ---------------------
        acreps_parser = subparsers.add_parser('ACREPS')
        acreps_parser.add_argument('--epsilon', type=float, default=0.1, help='KL constraint.')
        acreps_parser.add_argument('--gamma', type=float, default=0.98, help='discount factor γ.')

        acreps_parser.add_argument('--n_fourier', type=int, default=75, help='number of fourier features.')
        acreps_parser.add_argument('--fourier_band', type=float, nargs='+', help='number of fourier features.')

        # ---------------------
        #  PPO Arguments
        # ---------------------
        ppo_parser = subparsers.add_parser('PPO')
        ppo_parser.add_argument('--clip', type=float, default=0.2, help='clipping factor')
        ppo_parser.add_argument('--gamma', type=float, default=0.9, help='discount factor γ.')
        ppo_parser.add_argument('--gea', action='store_true', help='Use Generalized Advantage Estimation instead of '
                                                                   'Temporal Difference.')
        ppo_parser.add_argument('--lam', type=float, default=0.99, help='λ factor used by generalized advantage '
                                                                        'estimation.')
        ppo_parser.add_argument('--p_lr', type=float, default=7e-4, help='learning rate policy network')
        ppo_parser.add_argument('--v_lr', type=float, default=7e-4, help='learning rate value network')
        ppo_parser.add_argument('--mb_size', type=int, default=64, help='PPO mini-batch size')
        ppo_parser.add_argument('--n_mb_epochs', type=int, default=10,
                                help='number of epochs of PPO mini-batch optimization')

        args = parser.parse_args()

        if args.command is None:
            raise Exception('No algorithm specified! The first argument ')

        args.algorithm = args.command
        args.name = f"{args.algorithm}_{args.name}"
        del args.command
        return args
