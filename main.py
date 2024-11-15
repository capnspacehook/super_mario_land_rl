"""The main demo script includes yaml configs for all environments,
dynamic loading of environments, and other advanced features. If you
just want to run on a single environment, this is a simpler option."""

from pdb import set_trace as T
import argparse
from math import sqrt
import os

import numpy as np
import pufferlib
import pufferlib.models
import pufferlib.emulation
import pufferlib.vector
import pufferlib.frameworks.cleanrl
import torch as th
import wandb

from rich_argparse import RichHelpFormatter
from rich.console import Console
from rich.traceback import install

install(show_locals=False)

import clean_pufferl
from sweep import sweep

from super_mario_land.policy import Policy, Recurrent
import super_mario_land.settings
from register import createSMLEnv
from wrappers import VecRunningMean


def get_constants(module):
    # Get the module's global variables
    global_vars = module.__dict__

    # Filter out anything that is not a constant (no underscores, no imports)
    constants_dict = {
        name: value
        for name, value in global_vars.items()
        if not name.startswith("__") and not callable(value) and not isinstance(value, type(module))
    }

    return constants_dict


def make_policy(env, config):
    """Make the policy for the environment"""
    policy = Policy(env, config)
    policy = Recurrent(env, policy, config)
    return pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)


def train(args, shouldStopEarly=None):
    if args.track and args.mode != "sweep":
        args.wandb = init_wandb(args, args.wandb_name, id=args.train.exp_id)
        args.train.__dict__.update(dict(args.wandb.config.train))
    if args.vec.backend == "serial":
        backend = pufferlib.vector.Serial
    elif args.vec.backend == "multiprocessing":
        backend = pufferlib.vector.Multiprocessing
    elif args.vec == "ray":
        backend = pufferlib.vector.Ray
    else:
        raise ValueError(f"Invalid --vec.backend (serial/multiprocessing/ray).")

    evalVecenv = pufferlib.vector.make(
        createSMLEnv,
        env_args=(args.train,),
        env_kwargs=dict(isEval=True),
        backend=pufferlib.vector.Serial,
        num_envs=1,
    )
    evalInfos = []

    vecenv = pufferlib.vector.make(
        createSMLEnv,
        num_envs=args.vec.num_envs,
        env_args=(args.train,),
        env_kwargs=dict(render=args.render),
        num_workers=args.vec.num_workers,
        batch_size=args.vec.env_batch_size,
        zero_copy=args.vec.zero_copy,
        backend=backend,
    )
    policy = make_policy(vecenv.driver_env, args.train).to(args.train.device)

    data = clean_pufferl.create(args.train, vecenv, policy, wandb=args.wandb)

    try:
        bestEval = 0.0
        stopEarly = False
        nextEvalAt = args.train.eval_interval
        totalSteps = 0

        while data.global_step < args.train.total_timesteps:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)

            stepsTaken = data.global_step - totalSteps
            totalSteps = data.global_step
            if totalSteps + stepsTaken >= nextEvalAt:
                info, bestEval = eval_policy(evalVecenv, data.policy, data.config.device, data, bestEval)
                evalInfos.append(info)

                if shouldStopEarly is not None and shouldStopEarly(evalInfos, data):
                    stopEarly = True
                    break

                nextEvalAt += args.train.eval_interval
    except KeyboardInterrupt as e:
        clean_pufferl.close(data)
        raise e
    except Exception as e:
        Console().print_exception()
        clean_pufferl.close(data)
        raise e

    clean_pufferl.close(data)

    return evalInfos, stopEarly


def init_wandb(args, name, id=None, resume=True):
    wandb.init(
        id=id or wandb.util.generate_id(),
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        config={
            "train": dict(args.train),
            "vec": dict(args.vec),
            "env": get_constants(super_mario_land.settings),
        },
        name=name,
        save_code=True,
        resume=resume,
    )
    return wandb


def eval_policy(
    env: pufferlib.vector.Serial, policy, device, data=None, bestEval: float = None, printInfo=False
):
    steps = 0
    totalReward = 0.0

    state = None
    ob, _ = env.reset()
    driver = env.driver_env.env
    while True:
        with th.no_grad():
            ob = th.from_numpy(ob).to(device)
            if hasattr(policy, "lstm"):
                actions, value, state = policy.policy(ob, state)
            else:
                actions, value = policy.policy(ob)

            action = th.argmax(actions).cpu().numpy().reshape(env.action_space.shape)

        action_probs = actions.cpu().numpy().tolist()
        value = float(value)
        driver.recorder.setPolicyOutputs(action_probs, value)

        ob, reward, done, trunc, info = env.step(action)
        totalReward += reward
        steps += 1

        if printInfo:
            print(reward[0], action[0], action_probs, value)

        if done or trunc:
            break

    info = info[-1]

    if data is not None and data.wandb is not None:
        data.wandb.log(
            {
                "overview/agent_steps": data.global_step,
                "eval/reward": totalReward,
                "eval/length": steps,
                "eval/recording": wandb.Video("/tmp/eval.mp4"),
                **{f"eval/{k}": v for k, v in info.items()},
            }
        )

        score = info["progress"] + (100 / sqrt(steps))
        if bestEval is not None and score >= bestEval:
            bestEval = score
            clean_pufferl.save_checkpoint(data)
            data.msg = f"Checkpoint saved at update {data.epoch} for new best eval {bestEval}"

    info["reward"] = totalReward
    info["length"] = steps

    return info, bestEval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f":blowfish: PufferLib [bright_cyan]{pufferlib.__version__}[/]"
        " demo options. Shows valid args for your env and policy",
        formatter_class=RichHelpFormatter,
        add_help=False,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices="train eval evaluate playtest autotune sweep".split(),
    )
    parser.add_argument("--sweep-child", action="store_true")
    parser.add_argument("--eval-model-path", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--wandb-entity", type=str, default="capnspacehook", help="WandB entity")
    parser.add_argument("--wandb-project", type=str, default="Super Mario Land", help="WandB project")
    parser.add_argument("--wandb-group", type=str, default="", help="WandB group")
    parser.add_argument("--wandb-name", type=str, default="", help="WandB run name")
    parser.add_argument("--wandb-sweep", type=str, default="", help="Wandb sweep ID")
    parser.add_argument("--track", action="store_true", help="Track on WandB")

    # Train configuration
    parser.add_argument("--train.data-dir", type=str, default="checkpoints")
    parser.add_argument("--train.exp-id", type=str, default=None)
    parser.add_argument("--train.seed", type=int, default=-1)
    parser.add_argument("--train.torch-deterministic", action="store_true")
    parser.add_argument("--train.cpu-offload", action="store_true")
    parser.add_argument("--train.device", type=str, default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--train.total-timesteps", type=int, default=500_000_000)
    parser.add_argument("--train.checkpoint-interval", type=int, default=0)
    parser.add_argument("--train.eval-interval", type=int, default=1_000_000)
    parser.add_argument("--train.compile", action="store_true")
    parser.add_argument("--train.compile-mode", type=str, default="reduce-overhead")

    parser.add_argument("--train.batch-size", type=int, default=65_536)  # not swept
    parser.add_argument("--train.bptt-horizon", type=int, default=16)
    parser.add_argument("--train.clip-coef", type=float, default=0.2)  # not swept
    parser.add_argument("--train.clip-vloss", action="store_false")
    parser.add_argument("--train.ent-coef", type=float, default=0.0103695360320701)
    parser.add_argument("--train.gae-lambda", type=float, default=0.7897462671911262)
    parser.add_argument("--train.gamma", type=float, default=0.99)
    parser.add_argument("--train.learning-rate", type=float, default=0.0008491435467418811)
    parser.add_argument("--train.anneal-lr", action="store_true")
    parser.add_argument("--train.max-grad-norm", type=float, default=0.6716405153274536)
    parser.add_argument("--train.minibatch-size", type=int, default=16_384)  # not swept
    parser.add_argument("--train.norm-adv", action="store_false")
    parser.add_argument("--train.update-epochs", type=int, default=4)
    parser.add_argument("--train.vf-clip-coef", type=float, default=0.1)  # not swept
    parser.add_argument("--train.vf-coef", type=float, default=0.9445425476353746)
    parser.add_argument("--train.target-kl", type=float, default=0.2)  # not swept

    parser.add_argument("--train.game-area-embedding-dimensions", type=int, default=4)
    parser.add_argument("--train.cnn-filters", type=int, default=32)
    parser.add_argument("--train.entity-id-embedding-dimensions", type=int, default=4)
    parser.add_argument("--train.features-fc-hidden-units", type=int, default=256)
    parser.add_argument("--train.lstm-hidden-units", type=int, default=512)
    parser.add_argument("--train.actor-layers", type=int, default=1)
    parser.add_argument("--train.actor-hidden-units", type=int, default=512)
    parser.add_argument("--train.critic-layers", type=int, default=1)
    parser.add_argument("--train.critic-hidden-units", type=int, default=512)

    parser.add_argument("--train.reward-scale", type=float, default=0.004)
    parser.add_argument("--train.forward-reward", type=float, default=1.0)
    parser.add_argument("--train.wait-reward", type=float, default=0.25)
    parser.add_argument("--train.progress-reward", type=float, default=0.0)
    parser.add_argument("--train.backwards-punishment", type=float, default=0.0)
    parser.add_argument("--train.powerup-reward", type=float, default=10)
    parser.add_argument("--train.hit-punishment", type=float, default=-2.0)
    parser.add_argument("--train.heart-reward", type=float, default=10.0)
    parser.add_argument("--train.moving-platform-x-reward", type=float, default=0.15)
    parser.add_argument("--train.moving-platform-y-reward", type=float, default=1.25)
    parser.add_argument("--train.boulder_reward", type=float, default=5)
    parser.add_argument("--train.clear-level-reward", type=float, default=15)
    parser.add_argument("--train.death-punishment", type=float, default=-15)
    parser.add_argument("--train.game-over-punishment", type=float, default=-20)
    parser.add_argument("--train.coin-reward", type=float, default=3.0)
    parser.add_argument("--train.score-reward", type=float, default=0.005)
    parser.add_argument("--train.clock-punishment", type=float, default=0.0)

    parser.add_argument(
        "--vec.backend", type=str, default="multiprocessing", choices="serial multiprocessing ray".split()
    )
    parser.add_argument("--vec.num-envs", type=int, default=120)
    parser.add_argument("--vec.num-workers", type=int, default=24)
    parser.add_argument("--vec.env-batch-size", type=int, default=30)
    parser.add_argument("--vec.zero-copy", action="store_true")
    parsed = parser.parse_args()

    args = {}
    for k, v in vars(parsed).items():
        if "." in k:
            group, name = k.split(".")
            if group not in args:
                args[group] = {}

            args[group][name] = v
        else:
            args[k] = v

    args["train"] = pufferlib.namespace(**args["train"])
    args["vec"] = pufferlib.namespace(**args["vec"])
    args = pufferlib.namespace(**args)

    args.train.env = "sml"
    if args.train.seed == -1:
        args.train.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    if args.mode == "train":
        try:
            args.wandb = None
            train(args)
            if args.track:
                wandb.finish()
        except KeyboardInterrupt:
            os._exit(0)
        except Exception:
            Console().print_exception()
            os._exit(0)
    elif args.mode in ("eval", "evaluate"):
        try:
            env = pufferlib.vector.make(
                createSMLEnv,
                env_args=(args.train,),
                env_kwargs=dict(render=args.render, isEval=True, isInteractiveEval=True),
            )

            if args.eval_model_path is None:
                policy = make_policy(env, args.train).to(args.train.device)
            else:
                policy = th.load(args.eval_model_path, map_location=args.train.device)

            info = eval_policy(env, policy, args.train.device, printInfo=True)
            print(info)
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == "playtest":
        try:
            env = createSMLEnv(args.train, render=True, isPlaytest=True)
            obs, _ = env.reset()
            while True:
                obs, rew, term, trunc, _ = env.step(0)
                if term or trunc:
                    env.reset()
        except KeyboardInterrupt:
            env.close()
            os._exit(0)
    elif args.mode == "autotune":
        pufferlib.vector.autotune(createSMLEnv, batch_size=args.vec.env_batch_size)
    elif args.mode == "sweep":
        sweep(args, train)
