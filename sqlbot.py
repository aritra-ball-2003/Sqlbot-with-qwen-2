import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch 
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda:0'


st.title("PostgreSql-Bot")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="bfloat16",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def generate(conv):
    messages = [
        {"role": "system",
         "content": "Your task is to give solutions related to PostgreSql- its queries and commands"},
        {"role": "user", "content": conv}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output


file_up = st.file_uploader("Upload the code [2KB]")

with st.chat_message("assistant"):
    st.write("Hello Human!")

text_input = st.chat_input("Enter your code\n")

if text_input:
    with st.chat_message("user"):
        st.write(text_input)
    with st.chat_message("assistant"):
        response = generate(text_input)
        st.write(response)
    del model



if file_up:

    file = file_up.readlines()
    with st.chat_message("user"):
        st.write("File Uploaded")
    prompt = """Fix the following code, identify bottlenecks and suggest optimizations. Also highlight where changes have been made in the code :
        
    """
    print("\n\n\nFILE = \n\n\n",file,"\n\n\n")


    for line in file:
        if len(line.decode("utf-8")) > 0:
            prompt += line.decode('utf-8') + '\n'
    print(prompt)
    with st.chat_message("assistant"):
        response = generate(prompt)
        st.write(response)
    del model