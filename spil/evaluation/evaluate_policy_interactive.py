import argparse
from collections import Counter, defaultdict
import json
import logging
import os
from pathlib import Path
import sys
import time

# This is for using the locally installed repo clone when using slurm
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())
import hydra
import numpy as np
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from termcolor import colored
import torch
from tqdm.auto import tqdm

from calvin_env.envs.play_table_env import get_env
from spil.evaluation.multistep_sequences import get_sequences
from spil.evaluation.utils import get_default_model_and_env, get_env_state_for_initial_condition, join_vis_lang2, join_vis_lang
from spil.utils.utils import get_all_checkpoints, get_checkpoints_for_epochs, get_last_checkpoint

from spil.models.encoders.language_network import SBert

logger = logging.getLogger(__name__)


def make_env(dataset_path):
    val_folder = Path(dataset_path) / "validation"
    env = get_env(val_folder, show_gui=False)

    # insert your own env wrapper
    # env = Wrapper(env)
    return env


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[2] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
            os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


class CustomModel:
    def __init__(self):
        logger.warning("Please implement these methods as an interface to your custom model architecture.")
        raise NotImplementedError

    def reset(self):
        """
        This is called
        """
        raise NotImplementedError

    def step(self, obs, goal):
        """
        Args:
            obs: environment observations
            goal: embedded language goal
        Returns:
            action: predicted action
        """
        raise NotImplementedError


def count_success(results):
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        sr = n_success / len(results)
        step_success.append(sr)
    return step_success


def print_and_save(total_results, plan_dicts, args):
    log_dir = get_log_dir(args.log_dir)

    sequences = get_sequences(args.num_sequences)

    current_data = {}
    ranking = {}
    for checkpoint, results in total_results.items():
        epoch = checkpoint.stem.split("=")[1]
        print(f"Results for Epoch {epoch}:")
        avg_seq_len = np.mean(results)
        ranking[epoch] = avg_seq_len
        chain_sr = {i + 1: sr for i, sr in enumerate(count_success(results))}
        print(f"Average successful sequence length: {avg_seq_len}")
        print("Success rates for i instructions in a row:")
        for i, sr in chain_sr.items():
            print(f"{i}: {sr * 100:.1f}%")

        cnt_success = Counter()
        cnt_fail = Counter()

        for result, (_, sequence) in zip(results, sequences):
            for successful_tasks in sequence[:result]:
                cnt_success[successful_tasks] += 1
            if result < len(sequence):
                failed_task = sequence[result]
                cnt_fail[failed_task] += 1

        total = cnt_success + cnt_fail
        task_info = {}
        for task in total:
            task_info[task] = {"success": cnt_success[task], "total": total[task]}
            print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

        data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}

        current_data[epoch] = data

        print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    print(f"Best model: epoch {max(ranking, key=ranking.get)} with average sequences length of {max(ranking.values())}")

    for checkpoint, plan_dict in plan_dicts.items():
        epoch = checkpoint.stem.split("=")[1]

        ids, labels, plans, latent_goals = zip(
            *[
                (i, label, latent_goal, plan)
                for i, (label, plan_list) in enumerate(plan_dict.items())
                for latent_goal, plan in plan_list
            ]
        )
        latent_goals = torch.cat(latent_goals)
        plans = torch.cat(plans)
        np.savez(
            f"{log_dir / f'tsne_data_{epoch}.npz'}", ids=ids, labels=labels, plans=plans, latent_goals=latent_goals
        )


def evaluate_policy(model, env, sbert, args):
    conf_dir = Path(__file__).absolute().parents[2] / "conf"
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_sequences = get_sequences(args.num_sequences)

    results = []
    plans = defaultdict(list)

    # if not args.debug:
    #     eval_sequences = tqdm(eval_sequences, position=0, leave=True)

    for i in range(10):
        rollout(env, model, args, sbert)
        model.reset()

    return results, plans


def rollout(env, model, args, sbert):
    obs = env.get_obs()
    model.reset()
    lang_input = [input("What should I do? \n")]
    goal = {'lang': sbert(lang_input).squeeze(1)}

    for step in range(args.ep_len):
        action = model.step(obs, goal)
        obs, _, _, current_info = env.step(action)
        if args.debug:
            img_static, img_gripper = env.render(mode="rgb_static_gripper")
            join_vis_lang2(img_static, img_gripper, lang_input[0])
            time.sleep(3)
            # print('action: ', action)
        # check if current step solves a task


def main():
    seed_everything(0, workers=True)  # type:ignore
    parser = argparse.ArgumentParser(description="Evaluate a trained model on multistep sequences with language goals.")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset root directory.")

    # arguments for loading default model
    parser.add_argument(
        "--train_folder", type=str, help="If calvin_agent was used to train, specify path to the log dir."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=None,
        help="Comma separated list of epochs for which checkpoints will be loaded",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path of the checkpoint",
    )
    parser.add_argument(
        "--last_k_checkpoints",
        type=int,
        help="Specify the number of checkpoints you want to evaluate (starting from last). Only used for calvin_agent.",
    )

    parser.add_argument("--debug", action="store_true", help="Print debug info and visualize environment.")

    parser.add_argument("--log_dir", default=None, type=str, help="Where to log the evaluation results.")

    parser.add_argument("--device", default=0, type=int, help="CUDA device")
    args = parser.parse_args()

    # Do not change
    args.ep_len = 360
    args.num_sequences = 1000

    sbert = SBert('paraphrase-MiniLM-L3-v2')

    assert "train_folder" in args
    checkpoints = []
    if args.checkpoints is None and args.last_k_checkpoints is None and args.checkpoint is None:
        print("Evaluating model with last checkpoint.")
        checkpoints = [get_last_checkpoint(Path(args.train_folder))]
    elif args.checkpoints is not None:
        print(f"Evaluating model with checkpoints {args.checkpoints}.")
        checkpoints = get_checkpoints_for_epochs(Path(args.train_folder), args.checkpoints)
    elif args.checkpoints is None and args.last_k_checkpoints is not None:
        print(f"Evaluating model with last {args.last_k_checkpoints} checkpoints.")
        checkpoints = get_all_checkpoints(Path(args.train_folder))[-args.last_k_checkpoints:]
    elif args.checkpoint is not None:
        checkpoints = [Path(args.checkpoint)]

    env = None
    results = {}
    plans = {}
    model, env, _, _ = get_default_model_and_env(
        args.train_folder,
        args.dataset_path,
        checkpoints[-1],
        env=env,
        device_id=args.device,
    )
    evaluate_policy(model, env, sbert, args)

    print_and_save(results, plans, args)


if __name__ == "__main__":
    main()
