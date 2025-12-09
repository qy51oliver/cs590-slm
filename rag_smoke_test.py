import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipelines import apply_instruct_template

QUESTION = """When was the battle of Hastings?"""
CONTEXT = """“The Battle of Hastings was fought on 14 October 1066 between the Norman-French army of William, Duke of Normandy, and an English army under the Anglo-Saxon King Harold Godwinson, beginning the Norman Conquest of England. It took place approximately 7 mi (11 km) northwest of Hastings, close to the present-day town of Battle, East Sussex, and was a decisive Norman victory. The background to the battle was the death of the childless King Edward the Confessor in January 1066, which set up a succession struggle between several claimants to his throne. Harold was crowned king shortly after Edward's death but faced invasions by William, his own brother Tostig, and the Norwegian king Harald Hardrada (Harold III of Norway). Hardrada and Tostig defeated a hastily gathered army of Englishmen at the Battle of Fulford on 20 September 1066. They were in turn defeated by Harold at the Battle of Stamford Bridge on 25 September. The deaths of Tostig and Hardrada at Stamford Bridge left William as Harold's only serious opponent. While Harold and his forces were recovering, William landed his invasion forces in the south of England at Pevensey on 28 September and established a beachhead for his conquest...""""
# CONTEXT = ""

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