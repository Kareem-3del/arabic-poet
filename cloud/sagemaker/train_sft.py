# ruff: noqa: SIM115, E501, RUF005
"""QLoRA fine-tune of a Hugging Face chat model on an SFT set, inside a SageMaker training job.

Runs on one GPU (L40S 48 GB for a 32B model at 4-bit). Reads its settings from
/opt/ml/input/config/hyperparameters.json, the data from the "train" channel, and writes:
  /opt/ml/checkpoints   trainer checkpoints (synced to S3; a restarted spot job resumes here)
  /opt/ml/model         the PEFT adapter, training_log.json, samples.json, and (eval_prompts>0)
                        constrained_eval.json (uploaded at the end)

With eval_prompts>0 the tuned model is then benchmarked under the 'arud + vocabulary constraint
(arabic_poet.generation.constrained, shipped as poet_src.tar.gz by build_eval_bundle.py).
If an "adapter" channel holds a previous job's model.tar.gz, training is skipped and that
adapter is evaluated.

The assistant turn is rendered with an empty fast-thinking block (Jais-3 always opens one), so
at inference the prompt is the generation prompt at low reasoning effort plus the closing tag.
Loss is on the poem only.
"""
from __future__ import annotations

import json
import math
import os
import time

os.environ.setdefault("HF_HOME", "/opt/ml/cache/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

START = time.time()
HP_FILE = "/opt/ml/input/config/hyperparameters.json"
hp = json.load(open(HP_FILE)) if os.path.isfile(HP_FILE) else {}


def arg(name: str, default, cast=str):
    return cast(hp.get(name, default))


MODEL = arg("model_id", "inception42/Jais-3-32B")
REV = arg("revision", "c87efc1f6d94b1f5152ba1d2220aa68cdcb5f4ed")
DATA = arg("data_dir", "/opt/ml/input/data/train")
EPOCHS = arg("epochs", 1.0, float)
MAX_STEPS = arg("max_steps", -1, int)
LR = arg("lr", 1e-4, float)
BATCH = arg("batch", 2, int)
ACCUM = arg("accum", 4, int)
MAX_LEN = arg("max_len", 1200, int)
RANK = arg("rank", 16, int)
ALPHA = arg("alpha", 32, int)
TRAIN_LIMIT = arg("train_limit", 0, int)
BASE_SAMPLES = arg("base_samples", 0, int)
SAMPLES = arg("samples", 6, int)
EVAL_STEPS = arg("eval_steps", 200, int)
EVAL_PROMPTS = arg("eval_prompts", 0, int)  # constrained-decoding benchmark after training
EVAL_MODES = arg("eval_modes", "free,lexicon").split(",")
CODE = "/opt/ml/input/data/code"
ADAPTER_TAR = "/opt/ml/input/data/adapter/model.tar.gz"  # present => evaluate it, no training
EVAL_ONLY = os.path.isfile(ADAPTER_TAR)
THINK = arg("think_field", "think_faster")
CKPT, OUT = "/opt/ml/checkpoints", "/opt/ml/model"
os.makedirs(CKPT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def read(split: str) -> list[list[dict]]:
    with open(f"{DATA}/{split}.jsonl", encoding="utf-8") as f:
        return [json.loads(x)["messages"] for x in f if x.strip()]


train_msgs, valid_msgs, test_msgs = read("train"), read("valid"), read("test")
if TRAIN_LIMIT:
    train_msgs = train_msgs[:TRAIN_LIMIT]
print("data", len(train_msgs), len(valid_msgs), len(test_msgs), "| gpu", torch.cuda.get_device_name(0),
      flush=True)

tok = AutoTokenizer.from_pretrained(MODEL, revision=REV, trust_remote_code=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
CLOSE = f"</ifm|{THINK}>"
EFFORT = {"think": "high", "think_fast": "medium", "think_faster": "low"}[THINK]


def prompt_text(msgs: list[dict]) -> str:
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   reasoning_effort=EFFORT) + CLOSE


def encode(msgs: list[dict]):
    full = tok.apply_chat_template(msgs[:-1] + [{**msgs[-1], THINK: ""}], tokenize=False)
    prompt = prompt_text(msgs[:-1])
    assert full.startswith(prompt), "prompt is not a prefix of the full text"
    ids = tok(full, add_special_tokens=False)["input_ids"]
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    assert ids[: len(p_ids)] == p_ids, "prompt is not a token prefix of the full text"
    if len(ids) > MAX_LEN:
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * len(p_ids) + ids[len(p_ids):]}


def build(msgs):
    rows = [r for r in (encode(m) for m in msgs) if r]
    print(f"{len(rows)}/{len(msgs)} examples kept", flush=True)
    return Dataset.from_list(rows)


train_ds, valid_ds, test_ds = build(train_msgs), build(valid_msgs[:200]), build(test_msgs[:300])

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
t = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, revision=REV, trust_remote_code=True, quantization_config=bnb, device_map={"": 0},
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
print(f"model loaded in {time.time() - t:.0f}s | gpu mem {torch.cuda.memory_allocated() / 1e9:.1f} GB",
      flush=True)


def sample(n: int, label: str) -> list[dict]:
    model.eval()
    out = []
    for msgs in test_msgs[:n]:
        text = prompt_text(msgs[:-1])
        ids = tok(text, add_special_tokens=False, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**ids, max_new_tokens=320, do_sample=True, temperature=0.7,
                                 top_p=0.9, pad_token_id=tok.pad_token_id)
        poem = tok.decode(gen[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        out.append({"label": label, "request": msgs[-2]["content"], "poem": poem})
        print(f"--- {label}\n{msgs[-2]['content'][:200]}\n>>>\n{poem}\n", flush=True)
    return out


samples: list[dict] = sample(BASE_SAMPLES, "base") if BASE_SAMPLES else []
if samples:
    json.dump(samples, open(f"{OUT}/samples.json", "w"), ensure_ascii=False, indent=1)


def constrained_eval() -> None:
    import sys
    import tarfile

    tarfile.open(f"{CODE}/poet_src.tar.gz").extractall("/tmp/poet")
    sys.path.insert(0, "/tmp/poet")
    from arabic_poet.generation import constrained
    from arabic_poet.poetry import arud_fsa as fsa
    from arabic_poet.poetry.lexicon import Lexicon
    from arabic_poet.poetry.normalization import strip_diacritics

    lex = Lexicon(json.load(open(f"{CODE}/lexicon.json", encoding="utf-8")))
    bench = json.load(open(f"{CODE}/bench.json", encoding="utf-8"))[:EVAL_PROMPTS]
    stops = frozenset(tok.all_special_ids) | frozenset(tok.get_added_vocab().values())
    model.eval()
    items = []
    for it in bench:
        req = it["request"]
        ids = tok(prompt_text(it["messages"]), add_special_tokens=False)["input_ids"]
        row = {"id": it["id"], "request": req}
        for mode in EVAL_MODES:
            r = constrained.generate(
                model, tok, "", req["meter"], req["line_count"], req.get("rhyme"),
                constrain=mode != "free", seed=0, lexicon=lex if mode == "lexicon" else None,
                backend=constrained.TorchBackend(model), prompt_ids=ids, stop_ids=stops)
            abyat = [x.split("\n") for x in r.poem.split("\n\n")]
            abyat = [x for x in abyat if len(x) == 2]
            n = req["line_count"]
            exact = sum(fsa.complete(s, req["meter"], "sadr") + fsa.complete(a, req["meter"], "ajuz")
                        for s, a in abyat) / (2 * n)
            last = [strip_diacritics(a).split()[-1] if a.strip() else "" for _, a in abyat]
            rawi = (sum(1 for w in last if req["rhyme"] in w[-2:]) / n) if req.get("rhyme") else None
            row[mode] = {"poem": r.poem, "exact": round(exact, 3), "rawi": rawi,
                         "failed": r.failed_hemistichs, "retries": r.retries, "tokens": r.tokens,
                         "seconds": r.seconds, "log": r.log}
            print(f"=== {it['id']} {mode} exact {exact:.2f} rawi {rawi} failed {r.failed_hemistichs} "
                  f"retries {r.retries} {r.seconds}s\n{req.get('topic')} | {req['meter']} | "
                  f"{req.get('rhyme')}\n{r.poem}\n", flush=True)
        items.append(row)
        json.dump({"model": MODEL, "revision": REV, "modes": EVAL_MODES, "items": items},
                  open(f"{OUT}/constrained_eval.json", "w"), ensure_ascii=False, indent=1)


if EVAL_ONLY:
    import tarfile

    from peft import PeftModel

    tarfile.open(ADAPTER_TAR).extractall("/tmp/adapter")
    model = PeftModel.from_pretrained(model, "/tmp/adapter/adapter")
    print("loaded adapter from the adapter channel; training skipped", flush=True)
else:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    wanted = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    leaf = {n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    targets = [w for w in wanted if w in leaf]
    print("linear leaf names:", sorted(leaf)[:40], "| LoRA targets:", targets, flush=True)
    model = get_peft_model(model, LoraConfig(r=RANK, lora_alpha=ALPHA, lora_dropout=0.0, bias="none",
                                             target_modules=targets, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()


    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if logs and state.global_step % 50 == 0:
                print(f"step {state.global_step}/{state.max_steps} {logs} "
                      f"| {(time.time() - START) / 3600:.2f} h", flush=True)


    import dataclasses

    wanted_args = dict(
        output_dir=CKPT, num_train_epochs=EPOCHS, max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH, gradient_accumulation_steps=ACCUM,
        per_device_eval_batch_size=BATCH, learning_rate=LR, warmup_steps=50,
        lr_scheduler_type="cosine", optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, logging_steps=10, eval_strategy="steps", eval_steps=EVAL_STEPS,
        save_steps=EVAL_STEPS, save_total_limit=1, report_to="none", seed=0,
        dataloader_num_workers=2, group_by_length=True)
    known = {f.name for f in dataclasses.fields(TrainingArguments)}
    print("dropped (not in this transformers):", [k for k in wanted_args if k not in known], flush=True)
    args = TrainingArguments(**{k: v for k, v in wanted_args.items() if k in known})
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
                      callbacks=[Progress()])
    resume = any(d.startswith("checkpoint-") for d in os.listdir(CKPT))
    print("resuming from checkpoint" if resume else "fresh start", flush=True)
    trainer.train(resume_from_checkpoint=resume)

    model.save_pretrained(f"{OUT}/adapter")
    tok.save_pretrained(f"{OUT}/adapter")
    test_loss = trainer.evaluate(test_ds)["eval_loss"]
    log = {"model": MODEL, "revision": REV, "hours": round((time.time() - START) / 3600, 2),
           "test_loss": test_loss, "test_ppl": math.exp(test_loss), "targets": targets,
           "rank": RANK, "alpha": ALPHA, "lr": LR, "batch": BATCH * ACCUM, "max_len": MAX_LEN,
           "think_field": THINK, "train_examples": len(train_ds), "gpu": torch.cuda.get_device_name(0),
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
           "history": trainer.state.log_history}
    json.dump(log, open(f"{OUT}/training_log.json", "w"), ensure_ascii=False, indent=1)
    print({k: v for k, v in log.items() if k != "history"}, flush=True)

if SAMPLES:
    samples += sample(SAMPLES, "tuned")
    json.dump(samples, open(f"{OUT}/samples.json", "w"), ensure_ascii=False, indent=1)
if EVAL_PROMPTS:
    try:
        constrained_eval()
    except Exception:  # the adapter is already saved; report, do not lose the job
        import traceback

        traceback.print_exc()
        print("constrained eval failed", flush=True)
print("done", flush=True)
