import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid')



def estimator(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def calculate_pass_rates(directory):
    pattern = os.path.join(directory, "question_*_sample_*.json")
    files = glob.glob(pattern)
    questions = {}
    for file in files:
        with open(file, "r") as f:
            data = json.load(f)
        qid = data["task_id"]
        if qid > 120:
            continue
        if qid not in questions:
            questions[qid] = []
        questions[qid].append(data)
    num_samples_list = [len(samples) for samples in questions.values()]
    num_correct_list = [sum(1 for s in samples if s["correctness"]) for samples in questions.values()]
    overall_pass_rates = []
    for k in range(1, 33):
        rates = []
        for n, c in zip(num_samples_list, num_correct_list):
            rates.append(estimator(n, c, k))
        overall_pass = np.mean(rates) if rates else 0
        overall_pass_rates.append(overall_pass)
    return overall_pass_rates

def main():
    directories = [
        # "SkyThought/samples/taco-ll3-8B-llama-8b-big-lora-256-epochs-2-merged",
        # "SkyThought/samples/taco-ll3-8B-llama-8b-slerp-0.7-big-lora-256-epochs-2-merged",
        # "SkyThought/samples/taco-ll3-8B-llama-8b-big-slerp-0.5-lora-256-epochs-2-merged",
        # "SkyThought/samples/taco-ll3-8B-llama-8b-big-0.0-lora-256-epochs-2-merged",
        # "SkyThought/samples/Meta-Llama-3.1-8B-Instruct-Turbo"
        # "SkyThought/samples/s1_merge_slerp_0.7",
        # "SkyThought/samples/s1_merge_slerp_0.5",
        # "SkyThought/samples/s1_merge_slerp_0.9",
        # "SkyThought/samples/s1-32B",
        # "SkyThought/samples/checkpoint-68",
        # "SkyThought/samples/Qwen2.5-14B-Instruct",
        # "SkyThought/samples/taco-qwen25-14B-qwen25-14b-tacofull",
        # "SkyThought/samples/s1_merge_lerp_0.7",
        # "SkyThought/samples/s1_merge_lerp_0.5_diff_MEDIUM",
        # "SkyThought/samples/s1_merge_lerp_0.7_diff_MEDIUM",
        # "SkyThought/samples/s1_merge_lerp_0.9_diff_MEDIUM",
        # "SkyThought/samples/s1_merge_lerp_0.5_diff_HARD",
        # "SkyThought/samples/s1_merge_lerp_0.7_diff_HARD",
        # "SkyThought/samples/s1_merge_lerp_0.9_diff_HARD",
        # "SkyThought/samples/s1_merge_lerp_0.5",
        # "SkyThought/samples/s1-32B_diff_MEDIUM",
        # "SkyThought/samples/s1-32B_diff_HARD",
        # "SkyThought/samples/DeepSeek-R1-Distill-Llama-8B_diff_HARD",
        # "SkyThought/samples/DeepSeek-R1-Distill-Llama-8B_diff_MEDIUM",
        # "SkyThought/samples/llama-3.1-8b-0.7_diff_HARD",
        # "SkyThought/samples/llama-3.1-8b-0.7_diff_MEDIUM",
        # "SkyThought/samples/ll32_3b_lerp_0.7_diff_MEDIUM",
        # "SkyThought/samples/Llama-3.2-3B-Instruct_diff_MEDIUM",
        # "SkyThought/samples/ll32_1b_lerp_0.7_diff_MEDIUM",
        # "SkyThought/samples/Llama-3.2-1B-Instruct_diff_MEDIUM",
        "SkyThought/samples/llama-3.1-8b-0.7_diff_HARD",
        "SkyThought/samples/Llama-3.1-8B-Instruct_diff_MEDIUM",
        "SkyThought/samples/ll31_8b_lerp_0.7_diff_MEDIUM",
        # "SkyThought/samples/Qwen2.5-32B-Instruct_diff_HARD",
        # "SkyThought/samples/qwen25_32b_lerp_0.7_diff_HARD",
        "/scratch/gpfs/bs6865/messi-thinking/SkyThought/samples/ll31_8b_lerp_0.7_diff_MEDIUM_nosysprompt",
        "SkyThought/samples/Llama-3.1-8B-Instruct_diff_MEDIUM_nosysprompt"
    ]

    k_values = list(range(1, 33))
    plt.figure(figsize=(12, 8))
    
    names = {
        # "SkyThought/samples/taco-ll3-8B-llama-8b-big-lora-256-epochs-2-merged": r"$\alpha=1$",
        # "SkyThought/samples/taco-ll3-8B-llama-8b-slerp-0.7-big-lora-256-epochs-2-merged": r"$\alpha=0.7$",
        # "SkyThought/samples/taco-ll3-8B-llama-8b-big-slerp-0.5-lora-256-epochs-2-merged": r"$\alpha=0.5$",
        # "SkyThought/samples/taco-ll3-8B-llama-8b-big-0.0-lora-256-epochs-2-merged": r"$\alpha=0.0$",
        # "SkyThought/samples/Meta-Llama-3.1-8B-Instruct-Turbo": r"$\alpha=0.2$",
        "SkyThought/samples/s1_merge_slerp_0.7": r"$\alpha=0.7$ (s1)",
        "SkyThought/samples/s1_merge_slerp_0.5": r"$\alpha=0.5$ (s1)",
        "SkyThought/samples/s1_merge_slerp_0.9": r"$\alpha=0.9$ (s1)",
        "SkyThought/samples/s1-32B": r"$\alpha=1$ (s1)",
        "SkyThought/samples/checkpoint-68": r"$\alpha_{slerp}=0.7$",
        "SkyThought/samples/Qwen2.5-14B-Instruct": r"Instruct (non reasoning)",
        "SkyThought/samples/taco-qwen25-14B-qwen25-14b-tacofull": r"\alpha=1",
        "SkyThought/samples/s1_merge_lerp_0.7": r"$\alpha_{lerp}=0.7$ (s1)",
        "SkyThought/samples/s1_merge_lerp_0.5_diff_MEDIUM": r"$\alpha_{lerp}=0.5$ (s1, MEDIUM)",
        "SkyThought/samples/s1_merge_lerp_0.7_diff_MEDIUM": r"$\alpha_{lerp}=0.7$ (s1, MEDIUM)",
        "SkyThought/samples/s1_merge_lerp_0.9_diff_MEDIUM": r"$\alpha_{lerp}=0.9$ (s1, MEDIUM)",
        "SkyThought/samples/s1_merge_lerp_0.5": r"$\alpha_{lerp}=0.5$ (s1)",
        "SkyThought/samples/s1-32B_diff_MEDIUM": r"$\alpha=1$ (s1, MEDIUM)",
        "SkyThought/samples/s1-32B_diff_HARD": r"$\alpha=1$ (s1, HARD)",
        "SkyThought/samples/s1_merge_lerp_0.7_diff_HARD": r"$\alpha_{lerp}=0.7$ (s1, HARD)",
        "SkyThought/samples/s1_merge_lerp_0.9_diff_HARD": r"$\alpha_{lerp}=0.9$ (s1, HARD)",
        "SkyThought/samples/s1_merge_lerp_0.5_diff_HARD": r"$\alpha_{lerp}=0.5$ (s1, HARD)",
        "SkyThought/samples/DeepSeek-R1-Distill-Llama-8B_diff_HARD": r"$\alpha_{lerp}=1$ (Llama 8b distill, HARD)",
        "SkyThought/samples/DeepSeek-R1-Distill-Llama-8B_diff_MEDIUM": r"$\alpha_{lerp}=1$ (Llama 8b distill, MEDIUM)",
        "SkyThought/samples/llama-3.1-8b-0.7_diff_HARD": r"$\alpha_{lerp}=0.7$ (Llama 3.1, HARD)",
        "SkyThought/samples/llama-3.1-8b-0.7_diff_MEDIUM": r"$\alpha_{lerp}=0.7$ (Llama 3.1, MEDIUM)",
        "SkyThought/samples/Llama-3.2-3B-Instruct_diff_MEDIUM": r"$\alpha_{lerp}=1$ (Llama 3.2 3B, MEDIUM)",
        "SkyThought/samples/Llama-3.2-1B-Instruct_diff_MEDIUM": r"$\alpha_{lerp}=1$ (Llama 3.2 1B, MEDIUM)",
        "SkyThought/samples/ll32_3b_lerp_0.7_diff_MEDIUM": r"$\alpha_{lerp}=0.7$ (Llama 3.2 3B, MEDIUM)",
        "SkyThought/samples/ll32_1b_lerp_0.7_diff_MEDIUM": r"$\alpha_{lerp}=0.7$ (Llama 3.2 1B, MEDIUM)",
        "SkyThought/samples/llama-3.1-8b-0.7_diff_MEDIUM": r"$\alpha_{lerp}=0.7$ (Llama 3.1 8B, MEDIUM)",
        "SkyThought/samples/llama-3.1-8b-0.7_diff_HARD": r"$\alpha_{lerp}=0.7$ (Llama 3.1 8B, HARD)",
        "SkyThought/samples/DeepSeek-R1-Distill-Llama-8B_diff_HARD": r"$\alpha_{lerp}=1$ (Llama 8b distill, HARD)",
        "SkyThought/samples/DeepSeek-R1-Distill-Llama-8B_diff_MEDIUM": r"$\alpha_{lerp}=1$ (Llama 8b distill, MEDIUM)",
        "SkyThought/samples/ll31_8b_lerp_0.7_diff_MEDIUM": r"$\alpha_{lerp}=0.7$ (Llama 3.1 8B, MEDIUM)",
        "SkyThought/samples/Qwen2.5-32B-Instruct_diff_HARD": r"$\alpha_{lerp}=1$ (Qwen 32B, HARD)",
        "SkyThought/samples/qwen25_32b_lerp_0.7_diff_HARD": r"$\alpha_{lerp}=0.7$ (Qwen 32B, HARD)",
        "/scratch/gpfs/bs6865/messi-thinking/SkyThought/samples/ll31_8b_lerp_0.7_diff_MEDIUM_nosysprompt": r"$\alpha_{lerp}=0.7$ (Llama 3.1 8B, MEDIUM - NO SYS)",
        "SkyThought/samples/Llama-3.1-8B-Instruct_diff_MEDIUM_nosysprompt": r"$\alpha_{lerp}=1$ (Llama 3.1 8B, MEDIUM - NO SYS)"
    }
    
    for directory in directories:
        pass_rates = calculate_pass_rates(directory)
        label = names.get(directory, os.path.basename(os.path.normpath(directory)))
        # print it in a nice format
        print(f"{label}: {pass_rates}")
        sns.lineplot(x=k_values, y=pass_rates, marker="o", label=label, linewidth=3, markersize=10)
    
    plt.xlabel("k", fontsize=22)
    plt.ylabel("Pass@k", fontsize=22)
    plt.xticks(k_values, fontsize=22)
    plt.yticks(np.arange(0, 1, 0.1), fontsize=22)
    plt.ylim(0, 1)
    plt.grid(False)
    plt.legend(fontsize=12, frameon=False, loc="lower right")
    plt.tight_layout()
    sns.despine()
    plt.savefig("finetuning_pass_at_k_fulltaco_instruct2.pdf")
    plt.show()

if __name__ == "__main__":
    main() 