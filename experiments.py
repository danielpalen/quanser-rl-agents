import os
import yaml

import argparse

from joblib import Parallel, delayed

from run import run_single_experiment


def main():
    """
    This method is a helper function that is used to run a single algorithm multiple times.
    It can be configured and uses multiple processes for the different training runs so everything
    can run in parallel.

    It parses 3 required arguments from the command line.
    algo - the algorithm that should be used [REPS,ACREPS,PPO]
    name - name of the experiment
    env  - name of the environment that should be used.

    Upon first execution a file will be created located at ./out/experiments/[name]/parameters.yaml
    IMPORTANT: This file has to be filled with the parameters that are required for running the experiment.
    To figure out which parameters are required you can look at the required arguments for instantiating the
    desired algorithm, e.g. for REPS look at the REPS class and the require parameters to instantiate it.

    After preparing the yaml file the experiments.py command can be executed again and the experiments will run.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, required=True, help='algorithm to use [ACREPS,PPO]')
    parser.add_argument('--name', type=str, required=True, help='identifier to store experiment results')
    parser.add_argument('--env', type=str, required=True, help='name of the environment to be learned')
    parser.add_argument('--n_parallel', type=int, default=13, help='name of the environment to be learned')

    args = parser.parse_args()
    print(args)

    experiments_path = os.path.join('out', 'experiments', args.name)
    os.makedirs(experiments_path, exist_ok=True)

    # TODO: copy default file instead of creating empty one
    parameters_path = os.path.join(experiments_path, 'parameters.yaml')

    if not os.path.exists(parameters_path):
        with open(parameters_path, 'w') as params_file:
            args.n_experiments = 25
            yaml.dump({'params': vars(args)}, params_file, default_flow_style=False)
            print("HINT: In order to run experiments you have to add hyperparameter settings to the file"
                  f" ./{experiments_path}/parameters.yaml "
                  "containing the experiments hyperparameters. The folder structure has already been created for you")
    else:
        with open(parameters_path) as f:
            parameters = yaml.load(f)['params']

            def run_exp(i):
                args = parameters.copy()
                args['name'] = f"{args['name']}_{i}"
                args['summary_path'] = os.path.join(experiments_path, 'summary', args['name'])
                args['checkpoint_path'] = os.path.join(experiments_path, 'models', args['name'])
                os.makedirs(args['summary_path'], exist_ok=True)
                os.makedirs(args['checkpoint_path'], exist_ok=True)
                print(60 * '-' + '\n🤖 EXPERIMENT:', args['name'])
                run_single_experiment(args)

            Parallel(n_jobs=args.n_parallel)(delayed(run_exp)(experiment_no) for experiment_no in range(parameters['n_experiments']))


if __name__ == '__main__':
    main()
