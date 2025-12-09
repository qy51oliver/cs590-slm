import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipelines import apply_instruct_template

QUESTION = """What year was the battle of Hastings?"""
CONTEXT = "The Battle of Hastings was fought on 14 October 1066 between the Norman-French army of William, Duke of Normandy, and an English army under the Anglo-Saxon King Harold Godwinson, beginning the Norman Conquest of England. It took place approximately 7 mi (11 km) northwest of Hastings, close to the present-day town of Battle, East Sussex, and was a decisive Norman victory. The background to the battle was the death of the childless King Edward the Confessor in January 1066, which set up a succession struggle between several claimants to his throne. Harold was crowned king shortly after Edward's death but faced invasions by William, his own brother Tostig, and the Norwegian king Harald Hardrada (Harold III of Norway). Hardrada and Tostig defeated a hastily gathered army of Englishmen at the Battle of Fulford on 20 September 1066. They were in turn defeated by Harold at the Battle of Stamford Bridge on 25 September. The deaths of Tostig and Hardrada at Stamford Bridge left William as Harold's only serious opponent. While Harold and his forces were recovering, William landed his invasion forces in the south of England at Pevensey on 28 September and established a beachhead for his conquest..."

CONTEXT = """'[Battle of Hastings]\nThe Battle of Hastings was fought on 14 October 1066 between the Norman-French army of William, Duke of Normandy, and an English army under the Anglo-Saxon King Harold Godwinson, beginning the Norman Conquest of England. It took place approximately 7 mi (11 km) northwest of Hastings, close to the present-day town of Battle, East Sussex, and was a decisive Norman victory.\nThe background to the battle was the death of the childless King Edward the Confessor in January 1066, which set up a succession struggle between several claimants to his throne. Harold was crowned king shortly after Edward\'s death but faced invasions by William, his own brother Tostig, and the Norwegian king Harald Hardrada (Harold III of Norway). Hardrada and Tostig defeated a hastily gathered army of Englishmen at the Battle of Fulford on 20 September 1066. They were in turn defeated by Harold at the Battle of Stamford Bridge on 25 September. The deaths of Tostig and Hardrada at Stamford Bridge left William as Harold\'s only serious opponent. While Harold and his forces were recovering, William landed his invasion forces in the south of England at Pevensey on 28 September and established a beachhead for his conquest...\n\n---\n\n[William Hastings, 1st Baron Hastings]\nWilliam Hastings, 1st Baron Hastings (c. 1431 – 13 June 1483) was an English nobleman. A loyal follower of the House of York during the Wars of the Roses, he became a close friend and one of the most important courtiers of King Edward IV, whom he served as Lord Chamberlain. At the time of Edward\'s death he was one of the most powerful and richest men in England.  He was executed following accusations of treason by Edward\'s brother and ultimate successor, Richard III. The date of his death is disputed; early histories give 13 June, which is the traditional date.\n\n\n== Biography ==\n\nWilliam Hastings, born about 1430–1431, was the eldest son of Sir Leonard Hastings, and his wife Alice Camoys, daughter of Thomas de Camoys, 1st Baron Camoys.\nHastings succeeded his father in service to the House of York and through this service became close to his distant cousin the future Edward IV, whom he was to serve loyally all his life. He was High Sheriff of Warwickshire and High Sheriff of Leicestershire in 1455.\nHe fought alongside Edward at the Battle of Mortimer\'s Cross in Herefordshire in the Wars of the Roses, and was present at the proclamation of Edward as king in London on 4 March 1461, and then...\n\n---\n\n[Hastings]\nHastings (  HAY-stingz) is a seaside town and borough in East Sussex on the south coast of England,\n24 mi (39 km) east of Lewes and 53 mi (85 km) south east of London. The town gives its name to the Battle of Hastings, which took place 8 mi (13 km) to the north-west at Senlac Hill in 1066. It later became one of the medieval Cinque Ports. In the 19th century, it was a popular seaside resort, as the railway allowed tourists and visitors to reach the town. Hastings remains a popular seaside resort and is also a fishing port, with the UK\'s largest beach-based fishing fleet. The town\'s estimated population was 91,100 in 2021.\n\n\n== History ==\n\n\n=== Early history ===\nThe first mention of Hastings is from the late 8th century in the form Hastingas. This is derived from the Old English tribal name Hæstingas, meaning \'the constituency (followers) of Hæsta\'. Symeon of Durham records the victory of Offa in 771 over the Hestingorum gens, that is, "the people of the Hastings tribe." Hastingleigh in Kent was named after that tribe. The place name Hæstingaceaster is listed in the Anglo-Saxon Chronicle entry for 1050, and may be an alternative name for Hastings. However, the absence of any archaeological...\n\n---\n\n[Francis Rawdon-Hastings, 1st Marquess of Hastings]\nFrancis Edward Rawdon-Hastings, 1st Marquess of Hastings (9 December 1754 – 28 November 1826), styled The Honourable Francis Rawdon from birth until 1762, Lord Rawdon between 1762 and 1783, The Lord Rawdon from 1783 to 1793 and The Earl of Moira between 1793 and 1816, was an Anglo-Irish politician and military officer who served as Governor-General of Fort William from 1813 to 1823. He had also served with British forces for years during the American Revolutionary War and in 1794 during the War of the First Coalition. In Ireland, he was critical of the policy of coercion used to break the United Irish movement for representative government and national independence. He took the additional surname "Hastings" in 1790 in compliance with the will of his maternal uncle, Francis Hastings, 10th Earl of Huntingdon. \n\n\n== Background, education and early military career ==\nHastings was born at Moira, County Down, the son of John Rawdon, 1st Earl of Moira and Elizabeth Hastings, 13th Baroness Hastings, who was a daughter of the 9th Earl of Huntingdon. He was baptised at St. Audoen\'s Church, Dublin, on 2 January 1755. He grew up in Moira and in Dublin. He joined the British Army on 7 August 1771...\n\n---\n\n[The Battle of Hastings (album)]\nThe Battle of Hastings is the twelfth album by the British Canterbury scene progressive rock band Caravan, released in 1995.\n\n\n== Track listing ==\n"It\'s a Sad, Sad Affair" (Pye Hastings) – 3:23\n"Somewhere in Your Heart" (Hastings) – 5:42\n"Cold as Ice" (Hastings) – 4:09\n"Liar" (Hastings) – 6:07\n"Don\'t Want Love" (Pye Hastings, Jimmy Hastings) – 6:48\n"Travelling Ways" (Dave Sinclair) – 3:51\n"This Time" (Hastings) – 5:19\n"If It Wasn\'t for Your Ego" (Hastings) – 3:36\n"It\'s Not Real" (Hastings) – 5:29\n"Wendy Wants Another 6" Mole" (Hastings) – 2:25\n"I Know Why You\'re Laughing" (Hastings) – 5:32\n\n\n== Personnel ==\nCaravan\nPye Hastings – lead vocals, acoustic & electric guitar, Leslie guitar, harmony vocals\nGeoff Richardson – viola, violin, clarinet, acoustic & electric guitar, mandolin, tambourine, wind, kalimba, shaker, amplifiers, harmony vocals\nDave Sinclair – keyboards, harmony vocals\nJim Leverton – bass guitar, harmony vocals; lead vocals on "Travelling Ways"\nRichard Coughlan – drums\nAdditional personnel\nJimmy Hastings – flute, piccolo flute, alto flute, bass flute, clarinet, soprano sax, tenor sax\n\n\n== Releases information ==\n1999: Castle 520\n1996: CD HTD 41\n2002: CD Castle 72042\n1995: CD..."""

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