import pandas as pd
import streamlit as st
from langchain.memory import ConversationBufferMemory
from chatbot import resposta_bot, carrega_site
from langchain_groq import ChatGroq

arquivos_disponiveis = ['Conversa Livre', 'Site']
modelos_disponiveis = ['llama-3.3-70b-versatile', 'deepseek-r1-distill-llama-70b', 'llama-3.1-8b-instant']

st.header('Bem-Vindo ao Öyko!', divider=True)

def carrega_modelo(modelo, api_key):
    chat = ChatGroq(model=modelo, api_key=api_key)
    st.session_state['chat'] = chat

def pagina_chat():
    if 'memoria' not in st.session_state:
        st.session_state['memoria'] = ConversationBufferMemory()
    memoria = st.session_state['memoria']

    if 'chat' not in st.session_state:
        st.warning("Inicialize o modelo no menu lateral antes de começar a conversar.")
        st.warning('Seleção de modelos > Adicionar API Key > Inicializar Öyko')
        return

    chat_model = st.session_state['chat']
    tipo_arquivo = st.session_state.get('tipo_arquivo', 'Conversa Livre')
    documento = st.session_state.get('documento', '')

    for mensagem in memoria.buffer_as_messages:
        chat_msg = st.chat_message(mensagem.type)
        chat_msg.markdown(mensagem.content)

    input_user = st.chat_input("Fale com Öyko")
    if input_user:
        memoria.chat_memory.add_user_message(input_user)

        doc = "" if tipo_arquivo == "Conversa Livre" else documento
        resposta = resposta_bot(memoria.buffer_as_messages, doc, chat_model, input_user)

        memoria.chat_memory.add_ai_message(resposta)
        st.session_state['memoria'] = memoria

        st.chat_message("user").markdown(input_user)
        st.chat_message("assistant").markdown(resposta)


def sidebar():
    with st.sidebar:
        tabs = st.tabs(['Upload de arquivos', 'Seleção de modelo'])

        with tabs[0]:
            tipo_arquivo = st.selectbox('Escolha como deseja interagir com Öyko', arquivos_disponiveis)
            st.session_state['tipo_arquivo'] = tipo_arquivo

            if tipo_arquivo == 'Site':
                url_site = st.text_input('Cole o link do site')
                if url_site:
                    documento, aviso = carrega_site(url_site)
                    st.session_state['documento'] = documento
                    if aviso:
                        st.warning(aviso)

        with tabs[1]:
            modelo = st.selectbox('Escolha o modelo desejado', modelos_disponiveis)
            api_key = st.text_input(
                'Adicione sua API key da Groq para acessar o modelo',
                value=st.session_state.get(f'api_key_{modelo}', ''),
                type='password'
            )
            st.session_state[f'api_key_{modelo}'] = api_key
            st.session_state['modelo'] = modelo

        if st.button('Inicializar Öyko', use_container_width=True):
            if api_key.strip() == '':
                st.warning("Insira sua API Key para usar o Öyko")
            else:
                carrega_modelo(modelo, api_key)

def main():
    sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
