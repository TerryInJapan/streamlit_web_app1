#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS


# In[2]:


import os
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()


# In[3]:


docsearch = FAISS.load_local("Koto-Value-Sheet_index", embeddings)


# In[4]:


from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI


# In[5]:


from langchain.chat_models import ChatOpenAI
chain = load_qa_chain(ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff")

def generate_response(prompt):
    docs = docsearch.similarity_search(prompt,k=5)
    ans = chain.run(input_documents=docs, question=prompt)
    return ans


# In[6]:


import streamlit as st
from streamlit_chat import message


# In[7]:


# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated']=[]
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
def get_text():
    text_input = st.text_input("Enter your question")
    return text_input

user_input = get_text()

if user_input:
    output = generate_response(user_input)
    #store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)


# In[8]:


st.title("Chatbot for Koto Value Presentation")
st.caption("by T.Chiba May 30th,2023")


# In[12]:


if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i)+'_user')


# In[ ]:





# In[ ]:




