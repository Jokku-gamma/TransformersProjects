from transformers import AutoTokenizer,AutoModelForCausalLM
import gradio as gr

tokenizer=AutoTokenizer.from_pretrained("THUDM/GLM-Z1-9B-0414")
model = AutoModelForCausalLM.from_pretrained("THUDM/GLM-Z1-9B-0414")

def gen_text(prompt):
    input_ids=tokenizer.encode(prompt,return_tensors="pt").to(model.device)
    outputs=model.generate(input_ids,max_length=200,num_return_sequences=1,temperature=0.7)
    generated=tokenizer.decode(outputs[0],skip_special_tokens=True)
    return generated

interface=gr.Interface(
    fn=gen_text,
    inputs=gr.Textbox(lines=5,placeholder="Enter your prompt here .."),
    outputs=gr.Textbox(),
    title="GLM-Z1-9B-0414 text generation",
    description="Interact with GLM-Z1-9B-0414",
)
interface.launch()