import streamlit as st
import os
from teste import VideoFrameAnalyzer

# Prompts (reuse your current ones or set here)
def get_prompts():
    frame_prompt = f"""
    Você é um audiodescritor profissional certificado conforme normas brasileiras de acessibilidade
    (ANCINE - Instrução Normativa nº 145/2018 e nº 165/2022, Lei Brasileira de Inclusão nº 13.146/2015
    e ABNT NBR 17225).

    Sua tarefa é gerar a **audiodescrição profissional de um único frame** de um conteúdo informal,
    como um vídeo pessoal, meme, story ou gravação amadora.

    **Instruções:**
    1. Descreva apenas o que é visível no frame, sem supor o que aconteceu antes ou depois.
    2. Use linguagem descritiva neutra, mas acessível e natural, adequada para conteúdos informais.
    3. Evite termos excessivamente técnicos de cinema (como "plano americano" ou "plano detalhe"),
       mas mantenha precisão visual.
    4. Inclua elementos relevantes do contexto visual:
       - Pessoas, expressões, gestos, roupas, posições;
       - Objetos e cenário;
       - Texto presente na imagem (como legendas ou memes);
       - Emoções visuais perceptíveis (sem interpretar intenções);
       - Elementos de humor ou ironia, se explícitos visualmente.
    5. Coloque uma pequena descrição da interpretação da imagem, como "parece ser um meme sobre X" ou "a pessoa aparenta estar feliz".
        Mas evite interpratações longas.
    6. Se houver texto visível (por exemplo, legenda de meme), transcreva-o fielmente e indique sua posição.
    7. Mantenha clareza e concisão, como se fosse narrado para uma pessoa cega acompanhando o vídeo.

    **Formato de saída obrigatório (JSON):**
    {{
      "descricao_visual": "Texto objetivo e natural descrevendo o que aparece no frame, conforme normas brasileiras de audiodescrição adaptadas a conteúdo informal.",
      "elementos_relevantes": ["pessoa", "objeto", "texto_na_imagem", "expressao", "ambiente"],
      "observacoes_tecnicas": "Informações sobre iluminação, foco, tipo de imagem (selfie, print, meme, etc.)"
    }}
    """
    llama_prompt = f"""
    Você é um assistente especializado em gerar resumos narrativos de conteúdos audiodescritos.
    Você receberá **uma lista de audiodescrições em JSON**, cada uma descrevendo um frame de um vídeo, meme ou conteúdo informal.

    **Objetivo:** Gerar um **resumo coeso e compreensível**, combinando todas as descrições dos frames em narrativa clara e cronológica, mantendo fidelidade ao que é visível.  

    **Instruções:**
    1. Use linguagem clara, natural e neutra.
    2. Inclua elementos relevantes:
        - Pessoas, expressões, gestos, ações, roupas, objetos, cenários.
        - Textos visíveis em memes ou legendas.
        - Indicações de humor ou ironia se visualmente perceptíveis.
    3. Não invente informações ou inferências de eventos não descritos.
    4. Evite repetições desnecessárias.
    

    **Important:** Output **only** the final summary text or JSON. **Do not include introductions, explanations, disclaimers, or instructions.**

    **Lista de audiodescrições (entrada):**
    """
    return frame_prompt, llama_prompt

st.title("Análise de Imagem ou Vídeo com Llama")

uploaded_file = st.file_uploader("Faça upload de uma imagem ou vídeo", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to disk
    temp_path = os.path.join("temp_upload", uploaded_file.name)
    os.makedirs("temp_upload", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"Arquivo salvo em: {temp_path}")

    # Get prompts
    frame_prompt, llama_prompt = get_prompts()

    # Run analysis
    analyzer = VideoFrameAnalyzer()
    with st.spinner("Analisando arquivo, aguarde..."):
        try:
            result = analyzer.run(temp_path, "frames", 30, frame_prompt, llama_prompt)
            st.success("Análise concluída!")
            st.text_area("Resultado", result, height=300)
        except Exception as e:
            st.error(f"Erro ao analisar arquivo: {e}")

    # Optionally, clean up temp files after
    os.remove(temp_path)
    