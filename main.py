"""The main demo script includes yaml configs for all environments,
dynamic loading of environments, and other advanced features. If you
just want to run on a single environment, this is a simpler option."""

from pdb import set_trace as T
import argparse
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

from super_mario_land.policy import Policy
import super_mario_land.settings
from register import createSMLEnv


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


def make_policy(env, use_rnn):
    """Make the policy for the environment"""
    policy = Policy(env)
    if use_rnn:
        policy = Recurrent(env, policy)
        return pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        return pufferlib.frameworks.cleanrl.Policy(policy)


def train(args):
    args.wandb = None
    if args.track:
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

    vecenv = pufferlib.vector.make(
        createSMLEnv,
        num_envs=args.vec.num_envs,
        env_kwargs=dict(render=args.render),
        num_workers=args.vec.num_workers,
        batch_size=args.vec.env_batch_size,
        zero_copy=args.vec.zero_copy,
        backend=backend,
    )
    policy = make_policy(vecenv.driver_env, args.use_rnn).to(args.train.device)

    if args.train.seed == -1:
        args.train.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    args.train.env = "sml"
    data = clean_pufferl.create(args.train, vecenv, policy, wandb=args.wandb)

    evalVecenv = pufferlib.vector.make(
        createSMLEnv,
        env_kwargs=dict(isEval=True),
        backend=pufferlib.vector.Serial,
        num_envs=1,
    )

    try:
        nextEvalAt = args.train.eval_interval
        totalSteps = 0

        while data.global_step < args.train.total_timesteps:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)

            stepsTaken = data.global_step - totalSteps
            totalSteps = data.global_step
            if totalSteps + stepsTaken >= nextEvalAt:
                eval_policy(data, evalVecenv)
                nextEvalAt += args.train.eval_interval
    except KeyboardInterrupt:
        clean_pufferl.close(data)
        os._exit(0)
    except Exception:
        Console().print_exception()
        os._exit(0)

    clean_pufferl.close(data)


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
        monitor_gym=True,
        save_code=True,
        resume=resume,
    )
    return wandb


def eval_policy(data, env: pufferlib.vector.Serial):
    policy = data.policy
    device = data.config.device

    steps = 0
    totalReward = 0.0

    ob, _ = env.reset()
    while True:
        with th.no_grad():
            ob = th.from_numpy(ob).to(device)
            if hasattr(policy, "lstm"):
                # TODO: make this work deterministically
                action, _, _, _, state = policy(ob, state)
            else:
                action, _ = policy.policy(ob)

            action = th.argmax(action).cpu().numpy().reshape(env.action_space.shape)

        ob, reward, done, trunc, info = env.step(action)
        totalReward += reward
        steps += 1

        if done or trunc:
            break

    if data.wandb is not None:
        data.wandb.log(
            {
                "overview/agent_steps": data.global_step,
                "eval/reward": totalReward,
                "eval/length": steps,
                "eval/recording": wandb.Video("/tmp/eval.mp4"),
                **{f"eval/{k}": v for k, v in info[-1].items()},
            }
        )


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
        choices="train eval evaluate playtest autotune".split(),
    )
    parser.add_argument("--use-rnn", action="store_true")
    parser.add_argument("--eval-model-path", type=str, default=None, help="Path to model to evaluate")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--wandb-entity", type=str, default="capnspacehook", help="WandB entity")
    parser.add_argument("--wandb-project", type=str, default="Super Mario Land", help="WandB project")
    parser.add_argument("--wandb-group", type=str, default="", help="WandB group")
    parser.add_argument("--wandb-name", type=str, default="", help="WandB run name")
    parser.add_argument("--track", action="store_true", help="Track on WandB")

    # Train configuration
    parser.add_argument("--train.exp-id", type=str, default=None)
    parser.add_argument("--train.seed", type=int, default=-1)
    parser.add_argument("--train.torch-deterministic", action="store_true")
    parser.add_argument("--train.cpu-offload", action="store_true")
    parser.add_argument("--train.device", type=str, default="cuda" if th.cuda.is_available() else "cpu")
    parser.add_argument("--train.total-timesteps", type=int, default=100_000_000)
    parser.add_argument("--train.learning-rate", type=float, default=3e-05)
    parser.add_argument("--train.anneal-lr", action="store_true")
    parser.add_argument("--train.gamma", type=float, default=0.995)
    parser.add_argument("--train.gae-lambda", type=float, default=0.98)
    parser.add_argument("--train.update-epochs", type=int, default=5)
    parser.add_argument("--train.norm-adv", action="store_false")
    parser.add_argument("--train.clip-coef", type=float, default=0.2)
    parser.add_argument("--train.clip-vloss", action="store_true")
    parser.add_argument("--train.ent-coef", type=float, default=7e-03)
    parser.add_argument("--train.vf-coef", type=float, default=0.5)
    parser.add_argument("--train.vf-clip-coef", type=float, default=0.1)
    parser.add_argument("--train.max-grad-norm", type=float, default=1)
    parser.add_argument("--train.target-kl", type=float, default=None)
    parser.add_argument("--train.batch-size", type=int, default=98304)
    parser.add_argument("--train.minibatch-size", type=int, default=512)
    parser.add_argument("--train.bptt-horizon", type=int, default=16)
    parser.add_argument("--train.checkpoint-interval", type=int, default=0)
    parser.add_argument("--train.eval-interval", type=int, default=1_000_000)
    parser.add_argument("--train.compile", action="store_true")
    parser.add_argument("--train.compile-mode", type=str, default="reduce-overhead")

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

    if args.mode == "train":
        train(args)
    elif args.mode in ("eval", "evaluate"):
        try:
            clean_pufferl.rollout(
                createSMLEnv,
                env_kwargs={},
                agent_creator=make_policy,
                agent_kwargs={"use_rnn": args.use_rnn},
                model_path=args.eval_model_path,
                device=args.train.device,
            )
        except KeyboardInterrupt:
            os._exit(0)
    elif args.mode == "playtest":
        try:
            env = createSMLEnv(render=True, isPlaytest=True)
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
