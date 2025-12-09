import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipelines import apply_instruct_template

QUESTION = """When was the battle of Hastings?"""
CONTEXT = """Alexander Fleming discovered penicillin in 1928 at St Mary's Hospital in London."""
CONTEXT = ""

# -----------------------------------------

MODEL_NAME = "oliveryql/gemma270m-sft-fqa"


print(f"Loading {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# put on GPU if available, float16 there; otherwise CPU
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    trust_remote_code=True,
).eval()

q_text = QUESTION if not CONTEXT else f"{QUESTION}\nContext:\n{CONTEXT}"
messages = [
    {"role": "user", "content": f"Answer the question concisely.\nQuestion: {q_text}\nAnswer:"}
]

prompt = apply_instruct_template(messages, tokenizer)

# generate
enc = tokenizer([prompt], return_tensors="pt", padding=True).to(model.device)
with torch.no_grad():
    gen = model.generate(
        **enc,
        max_new_tokens=64,
        do_sample=False,          # deterministic
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
out = tokenizer.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

print("\n=== QUESTION ===")
print(QUESTION)
print("\n=== CONTEXT ===")
print(CONTEXT)
print("\n=== MODEL ANSWER ===")
print(out)