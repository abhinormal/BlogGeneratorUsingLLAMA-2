import streamlit as st
from langchain.prompts import PromptTemplate 
from langchain_community.llms import CTransformers  # Adjust the import based on the correct package

# Function to get response from llama 2 model
def getllamaresponse(input_text, no_words, blog_style):
    # LLAMA 2 MODEL
    llm = CTransformers(
        model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
        model_type='llama',
        config={'max_new_tokens': 256, 'temperature': 0.01}
    )

    # Prompt Template
    template = """
    Write a blog for a {blog_style} job profile on the topic "{input_text}"
    within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    formatted_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)

    # Generating the response from llama model
    try:
        response = llm(formatted_prompt)
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

    print(response)
    return response

st.set_page_config(page_title="Generate Blogs", page_icon='', layout='centered', initial_sidebar_state='collapsed')

st.header("Generate Blogs")

input_text = st.text_input("Enter the blog topic")


## creating two more columns for additional 2 fields

col1, col2=st.columns([5,5])

with col1:
    no_words=st.text_input('No of Words')
with col2:
    blog_style=st.selectbox('Writing the blog for', ('Researchers', 'Data Scientis', 'Common People'), index=0)
submit=st.button("Generate")

##Final response

if submit:
    st.write(getllamaresponse(input_text,no_words, blog_style))