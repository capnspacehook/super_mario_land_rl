from math import ceil, floor, log, log2
import logging
import os
import subprocess
import sys
import time

from rich.console import Console

from carbs import CARBS, CARBSParams, LinearSpace, LogitSpace, LogSpace, Param, WandbLoggingParams
import wandb
from wandb_carbs import WandbCarbs, create_sweep


logging.basicConfig(level=logging.INFO)


class CustomWandbCarbs(WandbCarbs):
    def __init__(self, carbs: CARBS, wandb_run=None):
        super().__init__(carbs, wandb_run)

    def _transform_suggestion(self, suggestion):
        print(f"Suggestion pre-transformation: {suggestion}")

        suggestion["batch_size"] = 2 ** suggestion["batch_size"]
        suggestion["bptt_horizon"] = 2 ** suggestion["bptt_horizon"]

        suggestion["total_timesteps"] = max(suggestion["total_timesteps"], 50_000_000)
        suggestion["update_epochs"] = min(suggestion["update_epochs"], 4)

        return suggestion

    def _suggestion_from_run(self, run):
        suggestion = super()._suggestion_from_run(run)

        suggestion["batch_size"] = int(log2(suggestion["batch_size"]))
        suggestion["bptt_horizon"] = int(log2(suggestion["bptt_horizon"]))

        return suggestion


def sweep(args, train):
    params = [
        Param(
            name="total_timesteps",
            space=LinearSpace(min=20_000_000, scale=50_000_000, rounding_factor=1_000_000, is_integer=True),
            search_center=20_000_000,
        ),
        # hyperparams
        Param(
            name="batch_size",
            space=LinearSpace(min=16, max=17, scale=1, is_integer=True),
            search_center=16,
        ),
        Param(
            name="bptt_horizon",
            space=LinearSpace(min=4, max=8, scale=3, is_integer=True),
            search_center=5,
        ),
        Param(name="ent_coef", space=LogSpace(scale=0.7), search_center=0.0075),
        Param(name="gae_lambda", space=LogitSpace(min=0.0, max=1.0), search_center=0.95),
        Param(name="gamma", space=LogitSpace(min=0.0, max=1.0), search_center=0.99),
        Param(name="learning_rate", space=LogSpace(scale=0.7), search_center=0.0001),
        Param(name="max_grad_norm", space=LinearSpace(min=0.0, scale=2.0), search_center=1.0),
        Param(
            name="update_epochs", space=LinearSpace(min=1, max=5, scale=3, is_integer=True), search_center=3
        ),
        Param(name="vf_coef", space=LogitSpace(min=0.0, max=1.0), search_center=0.3),
        # rewards
        # Param(name="forward_reward", space=LinearSpace(min=0.0, max=5.0, scale=2), search_center=1.0),
        # Param(name="powerup_reward", space=LinearSpace(min=0.0, max=20.0, scale=10), search_center=5.0),
        # Param(name="hit_punishment", space=LinearSpace(min=-15.0, max=0.0, scale=7), search_center=-2.0),
        # Param(name="heart_reward", space=LinearSpace(min=0.0, max=30.0, scale=10), search_center=10.0),
        # Param(name="clear_level_reward", space=LinearSpace(min=0.0, max=50.0, scale=10), search_center=15.0),
        # Param(name="death_punishment", space=LinearSpace(min=-50.0, max=0.0, scale=10), search_center=-15.0),
        # Param(name="coin_reward", space=LinearSpace(min=0.0, max=5.0), search_center=2.0),
        # Param(name="score_reward", space=LogSpace(), search_center=0.01),
    ]

    sweepID = args.wandb_sweep
    if not sweepID:
        sweepID = create_sweep(
            sweep_name=args.wandb_name,
            wandb_entity=args.wandb_entity,
            wandb_project=args.wandb_project,
            carb_params=params,
        )

    if args.sweep_child:
        try:
            trainWithSuggestion(args, params, train)
        except Exception:
            Console().print_exception()
        os._exit(0)

    def launchTrainingProcess():
        subprocess.run(args="docker container stop sml_postgres".split())
        subprocess.run(args="docker container rm sml_postgres".split())
        time.sleep(5)
        subprocess.run(
            args="docker run --name sml_postgres -d -e POSTGRES_PASSWORD=password -v sml_pgdata:/var/lib/postgresql/data -p 5432:5432 postgres:alpine -c max_connections=200".split()
        )
        time.sleep(5)

        childArgs = ["python", "main.py", "--mode=sweep", "--sweep-child", f"--wandb-sweep={sweepID}"]
        if args.train.compile:
            childArgs.append("--train.compile")
        print(f"running child training process with args: {childArgs}")
        ret = subprocess.run(args=childArgs, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)
        ret.check_returncode()

    wandb.agent(
        sweep_id=sweepID, entity=args.wandb_entity, project=args.wandb_project, function=launchTrainingProcess
    )


def trainWithSuggestion(args, params, train):
    args.track = False

    try:
        config = CARBSParams(
            seed=int(time.time()),
            better_direction_sign=1,
            max_suggestion_cost=14400,  # 4h
            num_random_samples=len(params),
            initial_search_radius=0.5,
            num_candidates_for_suggestion_per_dim=100,
            wandb_params=WandbLoggingParams(root_dir="wandb"),
        )
        carbs = CARBS(config=config, params=params)

        wandbCarbs = CustomWandbCarbs(carbs=carbs)
        resampling = not carbs._is_random_sampling() and len(carbs.success_observations) > (
            (carbs.resample_count + 1) * carbs.config.resample_frequency
        )
        print(
            f"loaded CARBS from wandb: success={len(carbs.success_observations)} failures={len(carbs.failure_observations)} outstanding={len(carbs.outstanding_suggestions)} resample_count={carbs.resample_count} resampling={resampling} random_sampling={carbs._is_random_sampling()} "
        )
        args.wandb = wandb

        suggestion = wandbCarbs.suggest()
        del suggestion["suggestion_uuid"]
        print(f"Suggestion: {suggestion}")

        args.train.__dict__.update(dict(suggestion))
        args.train.minibatch_size = args.train.batch_size // 4
        print(f"Training args: {args.train}")

        stopEarlyFunc = None
        if carbs._is_random_sampling():
            stopEarlyFunc = shouldStopEarly

        startTime = time.time()
        evalInfos, stoppedEarly = train(args, stopEarlyFunc)
        if stoppedEarly:
            wandbCarbs.record_early_stop()
            wandb.finish()
            return
    except Exception:
        Console().print_exception()
        wandbCarbs.record_failure()
        wandb.finish()
        return

    totalTime = time.time() - startTime
    bestEval = 0
    if len(evalInfos) != 0:
        bestEval = max(evalInfos, key=lambda i: i["progress"])["progress"]

    wandbCarbs.record_observation(objective=bestEval, cost=totalTime)
    wandb.finish()


# stop a sweep run early if the first level hasn't been completed in 45m
# and the run has more than 45m remaining or if the second level hasn't
# been completed in 1h30m and there is more than 45m remaining
def shouldStopEarly(infos, data):
    maxLevelCleared = max(infos, key=lambda i: i["levels_cleared"])["levels_cleared"]
    profile = data.profile

    return (profile.uptime >= 2700 and maxLevelCleared == 0 and profile.remaining > 2700) or (
        profile.uptime >= 5400 and maxLevelCleared == 1 and profile.remaining > 2700
    )


def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return int(2 ** min(possible_results, key=lambda z: abs(x - 2**z)))
