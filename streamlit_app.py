import streamlit as st
import openai

# Configuração da página
st.set_page_config(
    page_title="Funnel Mastermind AI",
    page_icon="🧠",
    layout="wide"
)

# Inicialização da API OpenAI
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Definição do sistema base de prompts
SYSTEM_PROMPT = """
Você é o Funnel Mastermind AI, um especialista em funis de vendas, copywriting e marketing digital.

Você foi criado por Glauco, um especialista em funis de vendas que está se posicionando como autoridade em funis perpétuos.

Seja específico, estratégico e utilize os princípios de Brevidade Inteligente: comunicação clara, direta e valiosa.

Use um tom consultivo profissional, direto e preciso, evitando linguagem genérica ou "coachzística".
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

# Função para consultar o modelo GPT-4o
def ask_gpt(prompt, system_prompt=SYSTEM_PROMPT):
    try:
        response = openai.chat.completions.create(
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

# Aviso sobre a versão
st.info(
    "Esta é uma versão inicial da Funnel Mastermind AI. " +
    "A funcionalidade de upload de documentos e base de conhecimento personalizada será implementada em breve."
)

# Criação das abas
tab1, tab2, tab3 = st.tabs([
    "💬 Chat com o Assistente", 
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
            response = ask_gpt(prompt)
            
            # Atualiza placeholder com resposta
            message_placeholder.markdown(response)
            
            # Adiciona resposta ao histórico
            st.session_state.messages.append({"role": "assistant", "content": response})

# Tab 2: Análise de Funis
with tab2:
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

# Tab 3: Criação de E-mails
with tab3:
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

# Rodapé
st.markdown("---")
st.markdown("**Funnel Mastermind AI** | Desenvolvido por Glauco | v2.0.0")
