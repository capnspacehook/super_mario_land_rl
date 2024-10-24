import json
from math import log, ceil, floor
import os
import subprocess
import sys
import time

from carbs import CARBS, CARBSParams, LinearSpace, LogitSpace, LogSpace, Param
import wandb
from wandb_carbs import WandbCarbs, create_sweep


def sweep(args, train):
    params = [
        Param(
            name="total_timesteps",
            space=LinearSpace(min=10_000_000, max=50_000_000, scale=10_000_000, is_integer=True),
            search_center=20_000_000,
        ),
        Param(name="learning_rate", space=LogSpace(min=1e-5, max=1e-1), search_center=3e-05),
        Param(name="gamma", space=LogitSpace(min=0.8, max=0.9999), search_center=0.995),
        Param(name="gae_lambda", space=LogitSpace(min=0.8, max=1.0), search_center=0.98),
        Param(
            name="update_epochs", space=LinearSpace(min=1, max=15, scale=5, is_integer=True), search_center=5
        ),
        Param(name="clip_coef", space=LogitSpace(min=0.1, max=0.4), search_center=0.2),
        Param(name="ent-coef", space=LogSpace(min=1e-5, max=1e-1), search_center=7e-03),
        Param(name="vf_coef", space=LogitSpace(min=0.0, max=1.0), search_center=0.5),
        Param(name="vf_clip_coef", space=LogitSpace(min=0.1, max=0.4), search_center=0.1),
        Param(name="max_grad_norm", space=LinearSpace(min=0, max=5), search_center=1),
        Param(
            name="batch_size",
            space=LinearSpace(min=65_536, max=1_048_576, scale=270_000, is_integer=True),
            search_center=98_304,
        ),
        Param(
            name="minibatch_size",
            space=LinearSpace(min=128, max=65_536, scale=20_000, is_integer=True),
            search_center=512,
        ),
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
        trainWithSuggestion(args, params, train)
        os._exit(0)

    def launchTrainingProcess():
        childArgs = ["python", "main.py", "--mode=sweep", "--sweep-child", f"--wandb-sweep={sweepID}"]
        if args.train.compile:
            childArgs.append("--train.compile")
        print(f"running child training process with args: {childArgs}")
        subprocess.run(args=childArgs, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

    wandb.agent(
        sweep_id=sweepID, entity=args.wandb_entity, project=args.wandb_project, function=launchTrainingProcess
    )


def trainWithSuggestion(args, params, train):
    wandb.init()
    args.track = False

    config = CARBSParams(
        better_direction_sign=1,
        max_suggestion_cost=10800,  # 3h
        num_random_samples=len(params),
        is_wandb_logging_enabled=False,
    )
    carbs = CARBS(config=config, params=params)

    wandbCarbs = WandbCarbs(carbs=carbs)
    print(
        f"Initialized CARBS with "
        + json.dumps(
            {
                "observations": wandbCarbs._num_observations,
                "failures": wandbCarbs._num_failures,
                "running": wandbCarbs._num_running,
                "defunct": wandbCarbs._defunct,
            }
        )
    )
    print(f"CARBS is random sampling: {carbs._is_random_sampling()}")
    args.wandb = wandb

    suggestion = wandbCarbs.suggest()
    del suggestion["suggestion_uuid"]
    args.train.__dict__.update(dict(suggestion))

    args.train.batch_size = closest_power(args.train.batch_size)
    args.train.minibatch_size = closest_power(args.train.minibatch_size)
    wandb.run.config.update(
        dict(
            batch_size=args.train.batch_size,
            minibatch_size=args.train.minibatch_size,
        ),
        allow_val_change=True,
    )
    print(f"Training args: {wandb.run.config}")

    startTime = time.time()
    try:
        evalInfos = train(args)
    except Exception:
        wandbCarbs.record_failure()
        wandb.finish()
        return

    totalTime = time.time() - startTime
    maxEvalLevelProgress = 0
    if len(evalInfos) != 0:
        maxEvalLevelProgress = max(evalInfos, key=lambda i: i["progress"])["progress"]

    wandbCarbs.record_observation(objective=maxEvalLevelProgress, cost=totalTime)
    wandb.finish()


def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return int(2 ** min(possible_results, key=lambda z: abs(x - 2**z)))
