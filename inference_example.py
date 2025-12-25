from system_prompts import TABC_LONG_PROMPT, TABC_PROMPT, GEN_PROMPT
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


###CHOOSE APPROPRIATE MODEL
model = "mehuldamani/hotpot-v2-brier-7b-no-split"   ## Refer to README.md for available models, note that classifier models are not supported in this file
##CHOOSE APPROPRIATE SYSTEM PROMPT
prompt_name = "TABC_LONG_PROMPT"                    ## Refer to README.md for corresponding system prompts

question = "Which popular dessert was invented at the Hungry Monk in Alfriston, Sussex?" 

if prompt_name == "TABC_LONG_PROMPT":
    sys_prompt = TABC_LONG_PROMPT
elif prompt_name == "TABC_PROMPT":
    sys_prompt = TABC_PROMPT 
elif prompt_name == "GEN_PROMPT":
    sys_prompt = GEN_PROMPT

user_format = (
                f"\n\nPROBLEM: {question}\n\n"
                )
prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_format},
            ]

tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
to_tokenize = [prompt]
prompt_ids = tokenizer.apply_chat_template(to_tokenize,add_generation_prompt=True)
texts = [tokenizer.decode(x) for x in prompt_ids]

sampling_params= SamplingParams(n = 1, temperature = 0, max_tokens=4096, seed=42) 
llm = LLM(model=model,gpu_memory_utilization=0.9)
outputs = llm.generate(texts,sampling_params=sampling_params)

print(outputs[0].outputs[0].text)