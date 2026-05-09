import ollama
import json
import os
from tqdm import tqdm

# --- Config ---
MODEL = "llama3.2"
TEMPERATURE = 0.9
N_TRIALS = 5
SAVE_PATH = "results.jsonl"
CUE_WORDS_PATH = "cue_words.txt"

# --- Load cue words from file ---
with open(CUE_WORDS_PATH, "r") as f:
    CUE_WORDS = [line.strip() for line in f if line.strip()]

# --- Load already completed cue/trial combinations if resuming ---
completed = set()
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "r") as f:
        for line in f:
            record = json.loads(line)
            completed.add((record["cue"], record["trial"]))

# --- Prompt function ---
def get_associations(cue, model, temperature):
    prompt = f"Generate the first 3 words that come to mind in response to the word {cue.upper()}. Respond with exactly 3 words separated by commas and nothing else."
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature}
    )
    raw = response["message"]["content"].strip()
    words = [w.strip().lower() for w in raw.split(",")]
    return words[:3]

# --- Main loop ---
with open(SAVE_PATH, "a") as f:
    for cue in tqdm(CUE_WORDS):
        for trial in range(1, N_TRIALS + 1):
            if (cue, trial) in completed:
                continue
            try:
                words = get_associations(cue, MODEL, TEMPERATURE)
                cleaned = True
            except Exception as e:
                print(f"Error on {cue}, trial {trial}: {e}")
                words = []
                cleaned = False

            record = {
                "cue": cue,
                "trial": trial,
                "response": words,
                "cleaned": cleaned
            }

            f.write(json.dumps(record) + "\n")
            f.flush()

print("Done.")