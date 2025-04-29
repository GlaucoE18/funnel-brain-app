import streamlit as st
import tempfile
import os
from supabase import create_client
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configurações
st.set_page_config(
    page_title="Funnel Mastermind AI",
    page_icon="🧠",
    layout="wide"
)

# Templates de prompts
QA_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em funis de vendas, copywriting e marketing digital.

Você foi criado por Glauco, um especialista em funis de vendas que está se posicionando como autoridade em funis perpétuos.

Seu conhecimento vem apenas de documentos específicos sobre marketing, funis e copywriting que você tem acesso.

Você deve responder às perguntas consultando apenas o contexto fornecido abaixo, sem usar conhecimento externo.

Seja específico, estratégico e utilize os princípios de Brevidade Inteligente: comunicação clara, direta e valiosa.

Se a resposta não estiver no contexto, diga que não tem essa informação específica no seu banco de conhecimento, mas pode ajudar com perguntas relacionadas a marketing e funis.

Use um tom consultivo profissional, direto e preciso, evitando linguagem genérica ou "coachzística".

Contexto:
{context}

Pergunta: {query}

Resposta:
"""

FUNNEL_ANALYSIS_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em funis de vendas.

Analise o funil descrito abaixo com base no seu conhecimento especializado. Avalie:

1. Estrutura do funil (Tipo de funil, fases, componentes)
2. Pontos fortes e oportunidades de melhoria
3. Estratégias de otimização recomendadas
4. Métricas que devem ser monitoradas

Use exemplos e referências do seu banco de conhecimento quando aplicável.

Descrição do funil:
{funnel_description}

Análise:
"""

EMAIL_F4_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em copywriting e e-mail marketing.

Crie um e-mail seguindo o framework F4 (Seinfeld + Brevidade Inteligente) com:

1. Assunto magnetizante que gera curiosidade
2. Abertura com gancho ou história que prende atenção
3. Ponte para o conteúdo principal
4. Conteúdo relevante e com valor prático
5. Call-to-action claro e persuasivo

O e-mail deve parecer pessoal, criar conexão, ter elementos de storytelling e seguir o princípio da Brevidade Inteligente.

Produto/Oferta: {offer}
Público-alvo: {audience}
Objetivo do e-mail: {objective}

E-mail:
"""

# Inicialização da conexão com Supabase e OpenAI
@st.cache_resource
def init_resources():
    # Conexão com Supabase
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase_client = create_client(supabase_url, supabase_key)
    
    # Configuração do OpenAI
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key
    )
    
    # Vector Store
    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embeddings,
        table_name="funnel_documents",
        query_name="match_documents"
    )
    
    # LLM
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        openai_api_key=openai_api_key
    )
    
    # QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_kwargs={"k": 5}
        ),
        chain_type_kwargs={
            "prompt": PromptTemplate.from_template(QA_PROMPT)
        },
        return_source_documents=True
    )
    
    return {
        "supabase_client": supabase_client,
        "embeddings": embeddings,
        "vector_store": vector_store,
        "llm": llm,
        "qa_chain": qa_chain
    }

# Inicializa recursos
try:
    resources = init_resources()
except Exception as e:
    st.error(f"Erro ao inicializar recursos: {str(e)}")
    st.stop()

# Função para processar documentos
def process_document(file, metadata=None):
    if metadata is None:
        metadata = {}
    
    # Cria arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    
    try:
        # Carrega PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        
        # Adiciona metadados
        for doc in documents:
            doc.metadata.update(metadata)
        
        # Divide em chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        # Adiciona ao vector store
        resources["vector_store"].add_documents(chunks)
        
        return {
            "success": True,
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "metadata": metadata
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
    finally:
        # Remove arquivo temporário
        os.unlink(temp_file_path)

# Função para fazer perguntas
def ask_question(question):
    response = resources["qa_chain"]({"query": question})
    return {
        "answer": response["result"],
        "sources": [doc.metadata for doc in response["source_documents"]]
    }

# Função para analisar funis
def analyze_funnel(description):
    prompt = FUNNEL_ANALYSIS_PROMPT.format(funnel_description=description)
    return ask_question(prompt)

# Função para criar e-mails
def create_email(offer, audience, objective):
    prompt = EMAIL_F4_PROMPT.format(
        offer=offer,
        audience=audience,
        objective=objective
    )
    return ask_question(prompt)

# Cabeçalho
st.title("🧠 Funnel Mastermind AI")
st.subheader("Seu assistente pessoal para funis de vendas, copywriting e marketing digital")

# Criação das abas
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat com o Assistente", 
    "📂 Upload de Documentos", 
    "🔍 Analise de Funis",
    "✉️ Criação de E-mails"
])

# Tab 1: Chat com o Assistente
with tab1:
    st.header("Converse com seu assistente especializado")
    
    # Inicializa histórico de chat se não existir
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Exibe histórico de mensagens
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Campo para nova mensagem
    if prompt := st.chat_input("Digite sua pergunta sobre funis, marketing, copywriting..."):
        # Adiciona mensagem do usuário ao histórico
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Exibe mensagem do usuário
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Exibe indicador de "pensando"
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Pensando...")
            
            # Obtém resposta
            try:
                response = ask_question(prompt)
                answer = response["answer"]
                
                # Exibe fontes se existirem
                if "sources" in response and response["sources"]:
                    answer += "\n\n**Fontes:**\n"
                    for source in response["sources"]:
                        if "title" in source:
                            answer += f"- {source['title']}\n"
                
                # Atualiza placeholder com resposta
                message_placeholder.markdown(answer)
                
                # Adiciona resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                message_placeholder.markdown(f"Erro ao processar sua pergunta. Por favor, tente novamente. Detalhes: {str(e)}")

# Tab 2: Upload de Documentos
with tab2:
    st.header("Adicione documentos à base de conhecimento")
    
    with st.form("upload_form", clear_on_submit=True):
        uploaded_file = st.file_uploader("Selecione um arquivo PDF", type=["pdf"])
        title = st.text_input("Título do documento", placeholder="Ex: Expert Secrets - Russell Brunson")
        author = st.text_input("Autor (opcional)", placeholder="Ex: Russell Brunson")
        category = st.text_input("Categoria (opcional)", placeholder="Ex: Copywriting, Funis, Email Marketing")
        
        submit_button = st.form_submit_button("Fazer Upload")
        
        if submit_button and uploaded_file is not None:
            with st.spinner("Processando documento..."):
                try:
                    # Prepara metadados
                    metadata = {
                        "title": title,
                        "filename": uploaded_file.name
                    }
                    
                    if author:
                        metadata["author"] = author
                    
                    if category:
                        metadata["category"] = category
                    
                    # Processa o documento
                    result = process_document(uploaded_file, metadata)
                    
                    if result.get("success", False):
                        st.success(f"Documento '{title}' processado com sucesso! Foram criados {result['chunk_count']} fragmentos de conhecimento.")
                    else:
                        st.error(f"Erro ao processar documento: {result.get('error', 'Erro desconhecido')}")
                
                except Exception as e:
                    st.error(f"Erro ao fazer upload: {str(e)}")

# Tab 3: Análise de Funis
with tab3:
    st.header("Analise um funil de vendas")
    
    description = st.text_area(
        "Descreva seu funil de vendas em detalhes",
        height=200,
        placeholder="Descreva seu funil de vendas em detalhes. Inclua etapas, produtos, pontos de conversão, e-mails, páginas, etc."
    )
    
    if st.button("Analisar Funil"):
        if description:
            with st.spinner("Analisando seu funil..."):
                try:
                    result = analyze_funnel(description)
                    st.markdown(result["answer"])
                    
                    # Exibe fontes se existirem
                    if "sources" in result and result["sources"]:
                        st.subheader("Fontes de conhecimento utilizadas:")
                        for source in result["sources"]:
                            if "title" in source:
                                st.write(f"- {source['title']}")
                
                except Exception as e:
                    st.error(f"Erro ao analisar funil: {str(e)}")
        else:
            st.warning("Por favor, forneça uma descrição do funil para análise.")

# Tab 4: Criação de E-mails
with tab4:
    st.header("Crie e-mails com o framework F4")
    
    with st.form("email_form"):
        offer = st.text_input("Produto ou Oferta", placeholder="Ex: Curso de Funis Perpétuos")
        audience = st.text_input("Público-alvo", placeholder="Ex: Designers que querem se tornar Funnel Builders")
        objective = st.text_input("Objetivo do e-mail", placeholder="Ex: Convidar para um webinar gratuito")
        
        submit_email = st.form_submit_button("Criar E-mail")
        
        if submit_email:
            if offer and audience and objective:
                with st.spinner("Criando seu e-mail..."):
                    try:
                        result = create_email(offer, audience, objective)
                        st.markdown(result["answer"])
                        
                        # Adiciona botão para copiar
                        st.download_button(
                            label="Baixar E-mail",
                            data=result["answer"],
                            file_name="email_f4.txt",
                            mime="text/plain"
                        )
                    
                    except Exception as e:
                        st.error(f"Erro ao criar e-mail: {str(e)}")
            else:
                st.warning("Por favor, preencha todos os campos.")

# Rodapé
st.markdown("---")
st.markdown("**Funnel Mastermind AI** | Desenvolvido por Glauco | v2.0.0")
