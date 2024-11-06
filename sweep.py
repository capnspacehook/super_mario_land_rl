from math import ceil, floor, log, log2, sqrt
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
        suggestion["bptt_horizon"] = closest_power(suggestion["bptt_horizon"])

        suggestion["features_fc_hidden_units"] = 2 ** suggestion["features_fc_hidden_units"]
        suggestion["lstm_hidden_units"] = 2 ** suggestion["lstm_hidden_units"]
        suggestion["actor_hidden_units"] = 2 ** suggestion["actor_hidden_units"]
        suggestion["critic_hidden_units"] = 2 ** suggestion["critic_hidden_units"]

        suggestion["game_over_punishment"] = (
            suggestion["death_punishment"] * suggestion["game_over_punishment"]
        )
        # negative values in log space isn't possible
        suggestion["clock_punishment"] = -suggestion["clock_punishment"]

        return suggestion

    def _suggestion_from_run(self, run):
        suggestion = super()._suggestion_from_run(run)

        suggestion["features_fc_hidden_units"] = int(log2(suggestion["features_fc_hidden_units"]))
        suggestion["lstm_hidden_units"] = int(log2(suggestion["lstm_hidden_units"]))
        suggestion["actor_hidden_units"] = int(log2(suggestion["actor_hidden_units"]))
        suggestion["critic_hidden_units"] = int(log2(suggestion["critic_hidden_units"]))

        suggestion["game_over_punishment"] = (
            suggestion["game_over_punishment"] / suggestion["death_punishment"]
        )
        # negative values in log space isn't possible
        suggestion["clock_punishment"] = abs(suggestion["clock_punishment"])

        return suggestion


def sweep(args, train):
    params = [
        Param(
            name="total_timesteps",
            space=LinearSpace(min=20_000_000, scale=50_000_000, is_integer=True),
            search_center=35_000_000,
        ),
        # hyperparams
        Param(
            name="bptt_horizon",
            space=LinearSpace(min=2, scale=50, is_integer=True),
            search_center=16,
        ),
        Param(name="ent_coef", space=LogSpace(min=0.0), search_center=0.0075),
        Param(name="gae_lambda", space=LogitSpace(min=0.0, max=1.0), search_center=0.95),
        Param(name="gamma", space=LogitSpace(min=0.0, max=1.0, scale=2), search_center=0.99),
        Param(name="learning_rate", space=LogSpace(min=0.0, scale=0.5), search_center=0.0001),
        Param(name="max_grad_norm", space=LinearSpace(min=0.0, scale=3.0), search_center=1.0),
        Param(
            name="update_epochs", space=LinearSpace(min=1, max=10, scale=3, is_integer=True), search_center=5
        ),
        Param(name="vf_coef", space=LogitSpace(min=0.0, max=1.0), search_center=0.3),
        # network arch
        Param(
            name="features_fc_hidden_units",
            space=LinearSpace(min=7, max=10, scale=3, is_integer=True),
            search_center=8,
        ),
        Param(
            name="features_fc_layers",
            space=LinearSpace(min=1, max=2, scale=1.5, is_integer=True),
            search_center=1,
        ),
        Param(
            name="lstm_hidden_units",
            space=LinearSpace(min=8, max=11, scale=3, is_integer=True),
            search_center=9,
        ),
        Param(
            name="actor_hidden_units",
            space=LinearSpace(min=8, max=11, scale=3, is_integer=True),
            search_center=9,
        ),
        Param(
            name="actor_layers",
            space=LinearSpace(min=1, max=2, scale=1.5, is_integer=True),
            search_center=1,
        ),
        Param(
            name="critic_hidden_units",
            space=LinearSpace(min=8, max=11, scale=3, is_integer=True),
            search_center=9,
        ),
        Param(
            name="critic_layers",
            space=LinearSpace(min=1, max=2, scale=1.5, is_integer=True),
            search_center=1,
        ),
        # rewards
        Param(name="reward_scale", space=LogSpace(min=0.0, max=1.0), search_center=0.004),
        Param(name="forward_reward", space=LinearSpace(min=0.0, max=5.0, scale=2), search_center=1.0),
        Param(name="backwards_punishment", space=LinearSpace(min=-5.0, max=0.0), search_center=-0.25),
        Param(name="powerup_reward", space=LinearSpace(min=0.0, max=20.0, scale=10), search_center=5.0),
        Param(name="hit_punishment", space=LinearSpace(min=-15.0, max=0.0, scale=5), search_center=-2.0),
        Param(name="heart_reward", space=LinearSpace(min=0.0, max=30.0, scale=10), search_center=10.0),
        Param(name="moving_platform_x_reward", space=LinearSpace(min=0.0, max=3.0), search_center=0.15),
        Param(name="moving_platform_y_reward", space=LinearSpace(min=0.0, max=3.0), search_center=1.25),
        Param(name="clear_level_reward", space=LinearSpace(min=0.0, max=50.0, scale=25), search_center=15.0),
        Param(name="death_punishment", space=LinearSpace(min=-50.0, max=0.0, scale=25), search_center=-15.0),
        Param(name="game_over_punishment", space=LinearSpace(min=1.0, max=2.0), search_center=1.0),
        Param(name="coin_reward", space=LinearSpace(min=0.0, max=5.0, scale=2), search_center=2.0),
        Param(name="score_reward", space=LogSpace(min=0.0, max=1.0), search_center=0.01),
        Param(name="clock_punishment", space=LogSpace(min=0.0, max=0.1), search_center=0.01),
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
            num_candidates_for_suggestion_per_dim=10,
            wandb_params=WandbLoggingParams(root_dir="wandb"),
        )
        carbs = CARBS(config=config, params=params)

        wandbCarbs = CustomWandbCarbs(carbs=carbs)
        resampling = carbs.config.resample_frequency > 0 and len(carbs.success_observations) > (
            (carbs.resample_count + 1) * carbs.config.resample_frequency
        )
        print(
            f"loaded CARBS from wandb: success={len(carbs.success_observations)} failures={len(carbs.failure_observations)} outstanding={len(carbs.outstanding_suggestions)} resample_count={carbs.resample_count} resampling={resampling} random_sampling={carbs._is_random_sampling()} "
        )
        args.wandb = wandb

        suggestion = wandbCarbs.suggest()
        del suggestion["suggestion_uuid"]
        print(f"Suggestion: {suggestion}")

        # validate suggestion
        if suggestion["features_fc_hidden_units"] > suggestion["lstm_hidden_units"]:
            wandbCarbs.record_failure()
            wandb.finish()
            return

        args.train.__dict__.update(dict(suggestion))
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
    runScore = 0
    if len(evalInfos) != 0:
        bestEval = max(evalInfos, key=lambda i: i["progress"])
        runScore = bestEval["progress"] + (100 / sqrt(bestEval["length"]))

    wandbCarbs.record_observation(objective=runScore, cost=totalTime)
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
