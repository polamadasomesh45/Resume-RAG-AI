import os
import hashlib

import streamlit as st
from dotenv import load_dotenv
from rag import ResumeRAG

load_dotenv()
st.set_page_config(page_title='ResumeRAG AI', page_icon='📄', layout='wide')
st.markdown('''<style>
.stApp {background: #0b1020; color: #eef2ff;}
.block-container {max-width: 1150px; padding-top: 2rem;}
[data-testid="stChatMessage"] {border: 1px solid #334155; border-radius: 14px;}
</style>''', unsafe_allow_html=True)

st.title('📄 ResumeRAG AI')
st.caption('Upload your resume • Ask evidence-grounded questions • Compare with a job description')

key = os.getenv('GEMINI_API_KEY')
if not key or key == 'replace_with_your_key':
    st.error('Add GEMINI_API_KEY to your .env file and restart Streamlit.')
    st.stop()

if 'rag' not in st.session_state:
    st.session_state.rag = ResumeRAG(key)
if 'messages' not in st.session_state:
    st.session_state.messages = []
rag = st.session_state.rag

with st.sidebar:
    st.header('1. Upload resume')
    uploaded = st.file_uploader('PDF or DOCX', type=['pdf', 'docx'])
    if uploaded and st.button('Process resume', type='primary', use_container_width=True):
        try:
            if uploaded.size > 5 * 1024 * 1024:
                raise ValueError('Please upload a file smaller than 5 MB.')
            data = uploaded.getvalue()
            with st.spinner('Extracting, embedding, and indexing...'):
                count = rag.index(data, uploaded.name)
            st.session_state.messages = []
            st.success(f'Indexed {count} chunks from {uploaded.name}')
        except Exception as exc:
            st.error(f'Processing failed: {exc}')
    st.divider()
    st.header('2. Optional job description')
    job_description = st.text_area('Paste job description', height=180)
    st.caption('Used as requirements for comparison, not as evidence about the candidate.')
    st.divider()
    if rag.filename:
        st.success(f'Active resume: {rag.filename}')
        st.write(f'Indexed chunks: {rag.collection.count()}')
    if st.button('Clear conversation'):
        st.session_state.messages = []
        st.rerun()
    if st.button('Remove resume and reset'):
        st.session_state.pop('rag', None)
        st.session_state.messages = []
        st.rerun()
    st.caption('Resume text is sent to Gemini for embeddings and answers. Avoid uploading sensitive resumes without consent.')

if not rag.filename:
    st.info('Upload and process a resume in the sidebar to begin.')
else:
    st.subheader('Ask about the resume')
    st.write('Try: “Summarize the projects”, “Which Python skills are evidenced?”, or “Compare this resume with the job description”.')

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        if message.get('hits'):
            with st.expander('Retrieved resume evidence'):
                for i, hit in enumerate(message['hits'], 1):
                    st.markdown(f'**Source {i} — page {hit["page"]}**')
                    st.code(hit['text'], language=None)

question = st.chat_input('Ask a question about the resume...', disabled=not bool(rag.filename))
if question:
    history = st.session_state.messages.copy()
    st.session_state.messages.append({'role': 'user', 'content': question})
    with st.chat_message('user'):
        st.markdown(question)
    with st.chat_message('assistant'):
        try:
            with st.spinner('Retrieving evidence and asking Gemini...'):
                answer, hits = rag.answer(question, history, job_description)
            st.markdown(answer)
            with st.expander('Retrieved resume evidence'):
                for i, hit in enumerate(hits, 1):
                    st.markdown(f'**Source {i} — page {hit["page"]}**')
                    st.code(hit['text'], language=None)
            st.session_state.messages.append({'role': 'assistant', 'content': answer, 'hits': hits})
        except Exception as exc:
            st.error(f'Unable to answer: {exc}')

if st.session_state.messages:
    transcript = '\n\n'.join(f'## {m["role"].title()}\n{m["content"]}' for m in st.session_state.messages)
    st.download_button('Download chat as Markdown', transcript, file_name='resume_chat.md', mime='text/markdown')
