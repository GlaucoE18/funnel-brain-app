import streamlit as st
import tempfile
import os
import time
from openai import OpenAI
from supabase import create_client
import pypdf
import uuid
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuração da página
st.set_page_config(
    page_title="Funnel Mastermind AI",
    page_icon="🧠",
    layout="wide"
)

# Inicialização da API OpenAI
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key)

# Inicialização do Supabase
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]
supabase = create_client(supabase_url, supabase_key)

# Configuração do OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key
)

# Definição do sistema base de prompts
SYSTEM_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em funis de vendas, copywriting e marketing digital.

Você foi criado por Glauco, um especialista em funis de vendas que está se posicionando como autoridade em funis perpétuos.

Seja específico, estratégico e utilize os princípios de Brevidade Inteligente: comunicação clara, direta e valiosa.

Use um tom consultivo profissional, direto e preciso, evitando linguagem genérica ou "coachzística".
"""

KNOWLEDGE_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em funis de vendas, copywriting e marketing digital.

Você foi criado por Glauco, um especialista em funis de vendas que está se posicionando como autoridade em funis perpétuos.

Responda à pergunta do usuário usando apenas as informações fornecidas no CONTEXTO abaixo. Se a resposta não estiver contida no CONTEXTO, diga que você não tem essa informação específica na sua base de conhecimento.

Seja específico, estratégico e utilize os princípios de Brevidade Inteligente: comunicação clara, direta e valiosa.

Use um tom consultivo profissional, direto e preciso, evitando linguagem genérica ou "coachzística".

CONTEXTO:
{context}

PERGUNTA:
{question}
"""

FUNNEL_ANALYSIS_PROMPT = """
Analise o funil descrito abaixo com base no seu conhecimento especializado. Avalie:

1. Estrutura do funil (Tipo de funil, fases, componentes)
2. Pontos fortes e oportunidades de melhoria
3. Estratégias de otimização recomendadas
4. Métricas que devem ser monitoradas

Descrição do funil:
"""

EMAIL_F4_PROMPT = """
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
"""

# Função para extrair texto de PDFs
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
    
    try:
        # Abre o PDF e extrai o texto
        reader = pypdf.PdfReader(temp_file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return text
    except Exception as e:
        st.error(f"Erro ao processar PDF: {str(e)}")
        return None
    finally:
        # Remove o arquivo temporário
        os.unlink(temp_file_path)

# Função para dividir texto em chunks
def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

# Função para criar embeddings e armazenar no Supabase
def store_embeddings(chunks, metadata):
    # Criar tabela se não existir
    supabase.table("funnel_documents").select("*").limit(1).execute()
    
    # Armazenar cada chunk
    for i, chunk in enumerate(chunks):
        # Gerar embedding via OpenAI
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding
        
        # Preparar metadados completos
        full_metadata = metadata.copy()
        full_metadata["chunk_index"] = i
        
        # Armazenar no Supabase
        supabase.table("funnel_documents").insert({
            "id": str(uuid.uuid4()),
            "content": chunk,
            "embedding": embedding,
            "metadata": full_metadata
        }).execute()
    
    return len(chunks)

# Função para buscar informações relevantes no Supabase
def search_knowledge_base(query, top_k=5):
    # Gerar embedding para a consulta
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Buscar documentos similares via função match_documents
    result = supabase.rpc(
        "match_documents", 
        {"query_embedding": query_embedding, "match_count": top_k}
    ).execute()
    
    if not result.data:
        return []
    
    # Formatar resultado
    contexts = []
    for item in result.data:
        contexts.append({
            "content": item["content"],
            "metadata": item["metadata"],
            "similarity": item["similarity"]
        })
    
    return contexts

# Função para consultar o modelo com base de conhecimento
def ask_with_knowledge(question):
    # Buscar informações relevantes
    contexts = search_knowledge_base(question)
    
    if not contexts:
        # Se não encontrar nada, use o prompt padrão
        return ask_gpt(question)
    
    # Preparar contexto para o prompt
    context_text = "\n\n---\n\n".join([c["content"] for c in contexts])
    
    # Consultar o modelo com o contexto
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": KNOWLEDGE_PROMPT.format(context=context_text, question=question)},
        ],
        temperature=0.7,
    )
    
    answer = response.choices[0].message.content
    
    # Adicionar fontes
    sources = []
    for context in contexts:
        if "title" in context["metadata"]:
            source = context["metadata"]["title"]
            if source not in sources:
                sources.append(source)
    
    if sources:
        answer += "\n\n**Fontes:**\n"
        for source in sources:
            answer += f"- {source}\n"
    
    return answer

# Função para consultar o modelo GPT-4o
def ask_gpt(prompt, system_prompt=SYSTEM_PROMPT):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao processar: {str(e)}"

# Função para analisar funis
def analyze_funnel(description):
    full_prompt = FUNNEL_ANALYSIS_PROMPT + description
    return ask_gpt(full_prompt)

# Função para criar e-mails
def create_email(offer, audience, objective):
    prompt = EMAIL_F4_PROMPT.format(
        offer=offer,
        audience=audience,
        objective=objective
    )
    return ask_gpt(prompt)

# Cabeçalho
st.title("🧠 Funnel Mastermind AI")
st.subheader("Seu assistente pessoal para funis de vendas, copywriting e marketing digital")

# Criação das abas
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Chat com o Assistente", 
    "📂 Upload de Documentos",
    "🔍 Análise de Funis",
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
            
            # Obtém resposta da OpenAI
            if "has_documents" in st.session_state and st.session_state.has_documents:
                response = ask_with_knowledge(prompt)
            else:
                response = ask_gpt(prompt)
            
            # Atualiza placeholder com resposta
            message_placeholder.markdown(response)
            
            # Adiciona resposta ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response})

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
                    # Extrai texto do PDF
                    text = extract_text_from_pdf(uploaded_file)
                    
                    if text:
                        # Divide em chunks
                        chunks = split_text(text)
                        
                        # Prepara metadados
                        metadata = {
                            "title": title,
                            "filename": uploaded_file.name
                        }
                        
                        if author:
                            metadata["author"] = author
                        
                        if category:
                            metadata["category"] = category
                        
                        # Armazena no Supabase
                        chunk_count = store_embeddings(chunks, metadata)
                        
                        # Marca que temos documentos
                        st.session_state.has_documents = True
                        
                        st.success(f"Documento '{title}' processado com sucesso! Foram criados {chunk_count} fragmentos de conhecimento.")
                    else:
                        st.error("Não foi possível extrair texto do documento.")
                
                except Exception as e:
                    st.error(f"Erro ao processar documento: {str(e)}")
    
    # Exibe documentos na base de conhecimento
    st.subheader("Documentos na base de conhecimento")
    
    try:
        result = supabase.table("funnel_documents").select("metadata").execute()
        
        if result.data:
            # Organiza documentos únicos por título
            documents = {}
            for item in result.data:
                if "metadata" in item and "title" in item["metadata"]:
                    title = item["metadata"]["title"]
                    if title not in documents:
                        documents[title] = item["metadata"]
            
            # Exibe lista de documentos
            if documents:
                st.write(f"Total de documentos: {len(documents)}")
                for title, metadata in documents.items():
                    author = metadata.get("author", "")
                    category = metadata.get("category", "")
                    info = f"**{title}**"
                    if author:
                        info += f" | Autor: {author}"
                    if category:
                        info += f" | Categoria: {category}"
                    st.markdown(info)
            else:
                st.info("Nenhum documento encontrado na base de conhecimento.")
        else:
            st.info("Nenhum documento encontrado na base de conhecimento.")
    
    except Exception as e:
        st.error(f"Erro ao carregar documentos: {str(e)}")

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
                result = analyze_funnel(description)
                st.markdown(result)
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
                    result = create_email(offer, audience, objective)
                    st.markdown(result)
                    
                    # Adiciona botão para copiar
                    st.download_button(
                        label="Baixar E-mail",
                        data=result,
                        file_name="email_f4.txt",
                        mime="text/plain"
                    )
            else:
                st.warning("Por favor, preencha todos os campos.")

# Verifica se existem documentos
@st.cache_data(ttl=300)
def check_documents():
    try:
        result = supabase.table("funnel_documents").select("id").limit(1).execute()
        return len(result.data) > 0
    except:
        return False

# Atualiza o estado da sessão
if "has_documents" not in st.session_state:
    st.session_state.has_documents = check_documents()

# Rodapé
st.markdown("---")
st.markdown("**Funnel Mastermind AI** | Desenvolvido por Glauco | v2.0.0")
