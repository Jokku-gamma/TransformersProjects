import streamlit as st
from transformers import pipeline
HF_TOKEN='YOUR_HUGGINGFACE_TOKEN'
MODELS={
    "distilgpt2":{
        "description":"Lightweight GPT2",
        "max_length":200
    },
    "gpt2":{
        "description":"Standard GPT2",
        "max_length":150
    }
}

def main():
    st.title("Text Gen ")
    selected_model=st.selectbox(
        "Choose a model",
        list(MODELS.keys()),
        format_func=lambda x : f"{x} ({MODELS[x]['description']})"
    )
    prompt=st.text_area("Your prompt")
    max_length=st.slider(
        "Max length",
        20,300,MODELS[selected_model]["max_length"]
    )
    temp=st.slider("Creativeness (0-1)",0.1,1.0,0.7)
    if st.button("Generate"):
        @st.cache_resource
        def load_model(model_name):
            return pipeline("text-generation",model=model_name,token=HF_TOKEN)
        with st.spinner(f"Loading {selected_model}"):
            gen=load_model(selected_model)
        with st.spinner("Generating .."):
            try:
                output=gen(
                    prompt,
                    max_length=max_length,
                    temperature=temp,
                    do_sample=True,
                    top_p=0.9
                )
                st.subheader("Generated Text")
                st.write(output[0]["generated_text"])
                st.markdown("---")
            except Exception as e:
                st.error(e)

if __name__=="__main__":
    main()