import os
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import WebBaseLoader
import re

def carrega_modelo(modelo, api_key):
    return ChatGroq(model=modelo, api_key=api_key)

def resposta_bot(mensagens, documento, chat, pergunta: str):
    try:
        msgs = [
            ("system", 
             "Você é o Öyko. Responda em PT-BR, de forma clara e curta. "
             "Se houver CONTEXTO abaixo, use-o. Não exiba <think> nem raciocínio oculto."),
        ]
        if documento:
            msgs.append(("system", "CONTEXTO:\n{documento}"))

        msgs.append(MessagesPlaceholder("history"))
        msgs.append(("user", "{pergunta}"))

        prompt = ChatPromptTemplate.from_messages(msgs)
        chain = prompt | chat
        resp = chain.invoke({
            "history": mensagens,
            "pergunta": pergunta,
            "documento": documento
        }).content

        return re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL).strip()
    except Exception as e:
        return f"Erro ao gerar resposta: {e}"


def carrega_site(url_site):
    aviso = ''
    documento = ''
    if url_site:
        try:
            loader = WebBaseLoader(url_site)
            lista_docs = loader.load()
            for doc in lista_docs:
                 documento += doc.page_content
            
            if len(documento) > 6000:
                        aviso = ("⚠️  O conteúdo do site é muito grande. Será cortado para caber no modelo.")
                        documento = documento[:6000]

            return documento, aviso
        except Exception as e:
            aviso = (f'Erro ao carregar o site: {e}')
            return documento, aviso
        
                

def conversa_livre(memoria, chat_model):
    documento = ''
    resposta = resposta_bot(memoria.buffer_as_messages, documento, chat_model)
    memoria.chat_memory.add_ai_message(resposta)
    return resposta


