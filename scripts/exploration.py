import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import csv
import math
import uuid
import argparse
from tqdm import tqdm
from tqdm.asyncio import tqdm
from collections import Counter
from vllm.lora.request import LoRARequest
from vllm import EngineArgs, LLMEngine, SamplingParams

from unstable.utils.templates import OBSERVATION_FORMATTING, extract_action_and_format_feedback

import wandb
import textarena as ta

def build_dataset(env_id, engine, sampling_params, lora_req, template, num_episodes=3):
    env=ta.make(env_id); env.reset(num_players=2); env.state.error_allowance=0
    observations = []
    for _ in tqdm(range(num_episodes), desc="Generating episodes"):
        turn = 0
        while True:
            turn += 1
            pid, obs = env.get_observation(); observations.append({"turn": turn, "id": len(observations), "pid": pid, "env_id": env_id, "observation": obs})
            print('LORA REQUEST', lora_req)
            engine.add_request(str(uuid.uuid4()), template(obs), sampling_params, lora_request=lora_req)
            action = None
            while action is None:
                responses = engine.step()
                for r in responses:
                    r = r.outputs[-1]
                    if r.finish_reason is not None:
                        action = extract_action_and_format_feedback(r.text)[0]
                        if action is None: engine.add_request(str(uuid.uuid4()), template(obs), sampling_params)
            done, _ = env.step(action)
            if done: break
        env.close()
    return observations


def rollout(example, engine, sampling_params, lora_req, template, num_turns: int = 25):
    prompt = template(example["observation"])
    actions = Counter(); request_ids = [str(uuid.uuid4()) for _ in range(num_turns)]; llm_responses = []
    print('LORA REQUEST', lora_req)
    for request_id in request_ids: engine.add_request(request_id, prompt, sampling_params, lora_request=lora_req)

    while sum(actions.values()) < num_turns:
        responses = engine.step()
        for r in responses:
            r = r.outputs[-1]
            if r.finish_reason is not None:
                llm_responses.append({"observation": example["observation"], "response": r.text})
                a = _extract_action(extract_action_and_format_feedback(r.text)[0])
                if a is not None: actions[a] += 1
                else: engine.add_request(str(uuid.uuid4()), prompt, sampling_params); request_ids.append(str(uuid.uuid4()))
    _write_data_to_file(llm_responses, f"{lora_req.lora_path if lora_req else 'default'}.csv")
    print('ACTION DISTRIBUTION ENV_ID', example["env_id"], actions)
    return _entropy(actions), actions


def evaluate(args, lora_path = None):
    engine_args = EngineArgs(
        model=args.model, enable_lora=True, max_loras=args.max_loras, max_lora_rank=args.lora_rank,
        max_cpu_loras=args.max_loras, max_num_seqs=args.max_parallel_seq, task="generate", max_model_len=args.max_model_len, 
        tensor_parallel_size=args.tensor_parallel_size, disable_custom_all_reduce=True, enforce_eager=False, disable_log_stats=True,  # Reduce logging overhead
    )
    try: engine = LLMEngine.from_engine_args(engine_args); print("VLLM engine initialized successfully")
    except Exception as e: print(f"vLLM engine initialization failed: {e}"); raise
    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    lora_req = LoRARequest(lora_path, 1, lora_path) if lora_path else None

    data = build_dataset(args.env_id, engine, sampling_params, lora_req, OBSERVATION_FORMATTING[args.template], args.num_episodes)
    metrics = {"all": {"entropy": [], "unique_actions": []}}
    for example in tqdm(data, desc="Evaluating examples"):
        entropy, actions = rollout(
            example,
            engine,
            sampling_params,
            lora_req,
            OBSERVATION_FORMATTING[args.template],
            args.num_turns
        )

        # store
        if example["turn"] not in metrics: metrics[example["turn"]] = {"entropy": [], "unique_actions": []}
        metrics[example["turn"]]["entropy"].append(entropy)
        metrics[example["turn"]]["unique_actions"].append(len(actions))
        metrics["all"]["entropy"].append(entropy)
        metrics["all"]["unique_actions"].append(len(actions))

    return {k: sum(v["entropy"]) / len(v["entropy"]) for k, v in metrics.items()}, {k: sum(v["unique_actions"]) / len(v["unique_actions"]) for k, v in metrics.items()}


def _extract_action(action: str, action_space = None): return (m.group(1).strip().lower() if (m := re.search(r"\[\s*(\d+)\s*\]", action)) else None)
def _entropy(actions: Counter) -> float: return -sum((c / sum(actions.values())) * math.log2(c / sum(actions.values())) for c in actions.values() if c > 0)
def _write_data_to_file(responses, filename: str):
    with open(filename, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['obs', 'response'])  # header
        for step in responses: writer.writerow([step["observation"], step["response"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="SimpleTak-v0-train")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--max_parallel_seq", type=int, default=40)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--checkpoints_dir", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=40)
    parser.add_argument("--max_loras", type=int, default=8)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--num_turns", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=2)
    parser.add_argument("--template", type=str, default="qwen3-zs")
    args = parser.parse_args()

    wandb.init(project="UnstableBaselines", config=vars(args), name=f"EXPLORATION-EVAL-{args.model}")

    checkpoint_paths = sorted(list(os.listdir(args.checkpoints_dir)), key=lambda x: int(x.split("-")[-1])) if args.checkpoints_dir else ["default"]
    if len(checkpoint_paths) > 1: checkpoint_paths = [cp for cp in checkpoint_paths if int(cp.split("-")[-1]) % args.eval_every == 0]
    for checkpoint in checkpoint_paths:
        print(f"Evaluating {args.model} with {checkpoint}")
        avg_entropy, avg_actions = evaluate(args, os.path.join(args.checkpoints_dir, checkpoint) if checkpoint != "default" else None)
        wandb.log({**{f"Turn {k}/avg entropy": v for k, v in avg_entropy.items()}, **{f"Turn {k}/avg actions": v for k, v in avg_actions.items()}})
