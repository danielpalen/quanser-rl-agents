
# python experiments.py --name test_experiment --n_experiments 25
import os
import yaml

from joblib import Parallel, delayed

from run import run_single_experiment


def main():

    name = 'ppo_trajreward'
    experiments_path = os.path.join('out', 'experiments', name)
    os.makedirs(experiments_path, exist_ok=True)

    # TODO: copy default file instead of creating empty one
    parameters_path = os.path.join(experiments_path, 'parameters.yaml')
    if not os.path.exists(parameters_path):
        open(parameters_path, 'a').close()

    try:
        with open(parameters_path) as f:
            parameters = yaml.load(f)['hyperparameters']
            #print(parameters)

            def run_exp(i):
                args = parameters.copy()
                args['name'] = f"{args['name']}_{i}"
                args['summary_path'] = os.path.join(experiments_path, 'summary', args['name'])
                args['checkpoint_path'] = os.path.join(experiments_path, 'models', args['name'])
                os.makedirs(args['summary_path'], exist_ok=True)
                os.makedirs(args['checkpoint_path'], exist_ok=True)
                print(60 * '-' + '\n🤖 EXPERIMENT:', args['name'])
                run_single_experiment(args)

            Parallel(n_jobs=8)(delayed(run_exp)(experiment_no) for experiment_no in range(parameters['n_experiments']))

            # for experiment_no in range(parameters['n_experiments']):
            #     args = parameters.copy()
            #     args['name'] = f"{args['name']}_{experiment_no}"
            #     args['summary_path'] = os.path.join(experiments_path, 'summary', args['name'])
            #     args['checkpoint_path'] = os.path.join(experiments_path, 'models', args['name'])
            #     os.makedirs(args['summary_path'], exist_ok=True)
            #     os.makedirs(args['checkpoint_path'], exist_ok=True)
            #
            #     print(60*'-'+'\n🤖 EXPERIMENT:', args['name'])
            #     run_single_experiment(args)

    except FileNotFoundError as e:
        print(e)
        print(f"HINT: In order to run experiments you have to create a file ./out/experiments/{name}/parameters.yaml'",
              "containing the experiments hyperparameters. The folder structure has already been created for you")


if __name__ == '__main__':
    main()
