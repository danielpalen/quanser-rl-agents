environments = {
    'balancer': {
        'sim':   'BallBalancerSim-v0',
        'robot': 'BallBalancerRR-v0',
    },

    'double_pendulum': {
        'sim':   'DoublePendulum-v0',
        'robot': 'DoublePendulumRR-v0',
    },

    'furuta': {
        'sim':   'Qube-v0',
        'robot': 'QubeRR-v0',
    },

    'pendulum': {
        'sim': 'Pendulum-v0',
    },

    'stab': {
        'sim':   'CartpoleStabLong-v0',
        # 'sim':   'CartpoleStabShort-v0',
        'robot': 'CartpoleStabShortRR-v0'
    },

    'swingup': {
        # 'sim':   'CartpoleSwingLong-v0',
        'sim':   'CartpoleSwingShort-v0',
        'robot': 'CartpoleSwingRR-v0'
    },
}


def get_env_name(env, sim=True):
    return environments[env]['sim' if sim else 'robot']
