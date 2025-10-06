import cv2
import base64
import json
from io import BytesIO
from PIL import Image
import os
import mimetypes
from huggingface_hub import InferenceClient

class VideoFrameAnalyzer:
    IMAGE_STATIC_PROMPT = f"""
    Você é um audiodescritor profissional certificado conforme normas brasileiras de acessibilidade
    (ANCINE - Instrução Normativa nº 145/2018 e nº 165/2022, Lei Brasileira de Inclusão nº 13.146/2015
    e ABNT NBR 17225).

    Sua tarefa é gerar a **audiodescrição profissional de uma imagem** de um conteúdo informal,
    como um meme, story ou quadrinho.

    **Instruções:**
    1. Descreva apenas o que é visível na imagem, sem supor o que aconteceu antes ou depois.
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
    """

    def image_analysis(self, image_path):
        print(f"Arquivo identificado como imagem: {image_path}")
        result = self.analyze_local_image(image_path, self.IMAGE_STATIC_PROMPT)
        # Salva resultado como se fosse um frame único
        frame_results = {image_path: result}
        with open("llama_frame_analysis.json", "w", encoding="utf-8") as f:
            json.dump(frame_results, f, indent=2, ensure_ascii=False)
        print("Resultado salvo em: llama_frame_analysis.json")
        llama_result = self.llama_call(self.IMAGE_STATIC_PROMPT)
        print("Resultado pós-processado pelo Llama:")
        print(llama_result)
        return llama_result

    def get_file_type_and_frame_count(self, file_path):
        """
        Returns (file_type, frame_count, fps, duration) where file_type is 'image' or 'video'.
        For images, frame_count, fps, duration are None.
        For videos, frame_count, fps, duration are ints/floats.
        """
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        ext = os.path.splitext(file_path)[1].lower()
        if ext in image_extensions:
            return 'image', None, None, None
        # Assume video otherwise
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise ValueError(f"Não foi possível abrir o arquivo: {file_path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return 'video', frame_count, fps, duration
    
    def __init__(self, api_key=None):
        self.client = InferenceClient(
            provider="nscale",
            api_key=api_key or "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"
        )

    def extract_frames(self, video_path, output_dir="frames", frame_interval=30):
        """
        Extrai frames do vídeo

        Args:
            video_path: Caminho para o arquivo de vídeo
            output_dir: Diretório para salvar os frames
            frame_interval: Intervalo entre frames (1 = todos os frames, 30 = a cada 30 frames)

        Returns:
            list: Lista com caminhos dos frames extraídos
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            # Remove all files in the output_dir
            for filename in os.listdir(output_dir):
                file_path = os.path.join(output_dir, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    import shutil
                    shutil.rmtree(file_path)

        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = f"{output_dir}/frame_{saved_count:06d}.jpg"
                cv2.imwrite(frame_filename, frame)
                frame_paths.append(frame_filename)
                saved_count += 1

            frame_count += 1

        cap.release()
        print(f"Extraídos {len(frame_paths)} frames do vídeo")
        return frame_paths



    def save_results(self, results, output_file="analysis_results.json"):
        """Salva resultados em arquivo JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Resultados salvos em: {output_file}")

    def encode_image_to_base64(self, image_path):
        """
        Converte uma imagem local para base64 data URL

        Args:
            image_path: Caminho para a imagem local

        Returns:
            str: Data URL da imagem em base64
        """
        # Verifica se o arquivo existe
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {image_path}")

        # Determina o tipo MIME da imagem
        mime_type, _ = mimetypes.guess_type(image_path)
        if mime_type is None or not mime_type.startswith('image/'):
            raise ValueError(f"Arquivo não é uma imagem válida: {image_path}")

        # Reduz o tamanho da imagem pela metade antes de codificar
        with Image.open(image_path) as img:
            width, height = img.size
            new_size = (max(1, width // 2), max(1, height // 2))
            img_resized = img.resize(new_size, Image.LANCZOS)
            buffer = BytesIO()
            img_resized.save(buffer, format=img.format)
            encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Retorna como data URL
        return f"data:{mime_type};base64,{encoded_string}"

    def analyze_local_image(self, image_path, prompt="Describe this image in one sentence.", api_key=None):
        """
        Analisa uma imagem local usando Hugging Face Inference

        Args:
            image_path: Caminho para a imagem local
            prompt: Texto do prompt para análise
            api_key: Chave da API do Hugging Face

        Returns:
            str: Resposta da análise
        """
    # Usa o cliente já instanciado

        try:
            # Converte a imagem para base64
            image_data_url = self.encode_image_to_base64(image_path)

            # Faz a requisição
            completion = self.client.chat.completions.create(
                model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url
                                }
                            }
                        ]
                    }
                ],
                temperature=0.5
            )

            return completion.choices[0].message.content

        except Exception as e:
            return f"Erro ao processar imagem: {str(e)}"


    def single_frame_analyses(self, image_paths, prompt="Describe this image in one sentence."):
        """
        Analisa múltiplas imagens locais

        Args:
            image_paths: Lista de caminhos para as imagens
            prompt: Prompt para análise

        Returns:
            dict: Dicionário com caminho da imagem como chave e análise como valor
        """
        results = {}

        for image_path in image_paths:
            print(f"Processando: {image_path}")
            try:
                result = self.analyze_local_image(image_path, prompt)
                results[image_path] = result
            except Exception as e:
                results[image_path] = f"Erro: {str(e)}"

        return results

    def llama_call(self, llama_prompt):
        with open("llama_frame_analysis.json", "r", encoding="utf-8") as f:
            data = json.load(f)

        json_text = json.dumps(data, indent=2, ensure_ascii=False)
        completion = self.client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": llama_prompt},
                        {"type": "text", "text": json_text}
                    ]
                }
            ],
            temperature=0.5
        )
        return completion.choices[0].message.content

    def run(self, file_path, output_dir, frame_interval, frame_prompt, llama_prompt):
        file_type, frame_count, fps, duration = self.get_file_type_and_frame_count(file_path)
        if file_type == 'image':
            return self.image_analysis(file_path)
        elif file_type == 'video':
            print(f"Arquivo identificado como vídeo: {file_path}")
            print(f"Frames: {frame_count}, FPS: {fps}, Duração: {duration:.2f}s")
            # Ajusta frame_interval para sempre pegar 15 frames
            if frame_count < 15:
                frame_interval = 1
            else:
                frame_interval = max(1, int(frame_count // 15))
            print(f"Intervalo ajustado para {frame_interval} para extrair 15 frames.")
            frame_paths = self.extract_frames(file_path, output_dir, frame_interval)
            frame_results = self.single_frame_analyses(frame_paths, frame_prompt)
            with open("llama_frame_analysis.json", "w", encoding="utf-8") as f:
                json.dump(frame_results, f, indent=2, ensure_ascii=False)
            print("Resultados salvos em: llama_frame_analysis.json")
            result = self.llama_call(llama_prompt)
            print(result)
            return result
        else:
            raise ValueError("Tipo de arquivo não suportado.")

# Main function to run the analysis and save results


if __name__ == "__main__":
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

    analyzer = VideoFrameAnalyzer()
    analyzer.run("helo.mp4", "frames", 30, frame_prompt, llama_prompt)