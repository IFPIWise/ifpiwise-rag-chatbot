import ollama
import gradio as gr
import os
import glob
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from chromadb.config import Settings
from chromadb import Client, PersistentClient
from langchain_community.vectorstores import Chroma
from tqdm import tqdm
from langdetect import detect

# Hiperparâmetros e constantes
PDF_DIR = "docs"
CHROMADB_DIR = "chromadb"  # Diretório para persistência do ChromaDB
MODELO_LLM = "llama2"
COLLECTION_NAME = "chaves"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MAX_WORKERS = 16
LLM_TEMPERATURE = 0.005
NUM_DOCUMENTS = 4  # Número de documentos a recuperar
SEARCH_TYPE = "mmr"  # Tipos: 'similarity', 'mmr'
FETCH_K = 20  # Número de documentos para considerar antes de aplicar MMR
# Parâmetro lambda para MMR (0-1): 0 = diversidade máxima, 1 = relevância máxima
LAMBDA_MULT = 0.5

# Prompt do sistema
SYSTEM_PROMPT = """Você é o assistente virtual oficial do Instituto Federal do Piauí (IFPI), especializado em documentos normativos institucionais como resoluções, regulamentos, portarias e outros documentos oficiais. 
Com base no contexto fornecido, responda à consulta do usuário de forma clara, objetiva e formal, seguindo estas diretrizes:

1. Responda APENAS com informações contidas nos documentos normativos fornecidos;
2. Caso a resposta não esteja presente no contexto, responda 'Esta informação não consta nos documentos normativos disponíveis';
3. Mencione, sempre que possível, a fonte específica da informação (número da resolução, artigo, etc.);
4. Mantenha um tom institucional, formal e preciso, adequado ao ambiente acadêmico;
5. Não faça suposições além do conteúdo dos documentos;
6. Se houver conflito entre normas, indique as diferentes interpretações presentes nos documentos;
7. Seja conciso e direto, mas completo ao abordar questões procedimentais.
8. Sempre responda em português.
"""

EXAMPLES = [
    "Qual a missão institucional do IFPI?",
    "O que é o Conselho de Classe?"
]

# Configurações da interface
UI_TITLE = "RAG Chatbot IFPI"
UI_DESCRIPTION = "Faça uma pergunta sobre qualquer documento na pasta 'docs'"

# Verifica se a pasta docs existe
if not os.path.exists(PDF_DIR):
    raise FileNotFoundError(f"A pasta '{PDF_DIR}' não foi encontrada.")

# Verifica se existem PDFs na pasta docs
pdf_files = glob.glob(f"{PDF_DIR}/*.pdf")
if not pdf_files:
    raise FileNotFoundError(
        f"Nenhum arquivo PDF encontrado na pasta '{PDF_DIR}'.")


def inicializar_chromadb():
    if os.path.exists(CHROMADB_DIR):
        usar_existente = input(
            "Base de dados existente encontrada. Deseja reutilizá-la? (s/n): ").lower()
        if usar_existente == 's':
            print("Usando base de dados existente...")
            client = PersistentClient(path=CHROMADB_DIR)
            return client.get_collection(name=COLLECTION_NAME), True
        else:
            print("Recriando base de dados...")
            import shutil
            shutil.rmtree(CHROMADB_DIR)

    os.makedirs(CHROMADB_DIR, exist_ok=True)
    client = PersistentClient(path=CHROMADB_DIR)
    collection = client.create_collection(name=COLLECTION_NAME)
    return collection, False


# Verifica se o Ollama está rodando e se o modelo Llama2 está disponível
try:
    ollama.pull(MODELO_LLM)  # Baixa o modelo
except Exception as e:
    print(f"Erro ao conectar ao Ollama: {e}")
    print("Certifique-se de que o Ollama está instalado e rodando.")
    exit(1)

# Função para carregar documentos de um arquivo PDF


def load_document(pdf_path):
    print(f"Carregando documento: {pdf_path}")
    loader = PyMuPDFLoader(pdf_path)
    return loader.load()


# Initialize Chroma client and create/reset the collection
collection, usando_existente = inicializar_chromadb()

if not usando_existente:
    # Carregar e processar documentos apenas se não estiver usando base existente
    all_documents = []
    for pdf_file in pdf_files:
        try:
            documents = load_document(pdf_file)
            all_documents.extend(documents)
            print(f"Carregado {len(documents)} páginas de {pdf_file}")
        except Exception as e:
            print(f"Erro ao carregar {pdf_file}: {e}")

    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(all_documents)
    print(f"Total de {len(chunks)} chunks criados de todos os documentos")

    # Initialize Ollama embeddings
    embedding_function = OllamaEmbeddings(model=MODELO_LLM)

    def generate_embedding(chunk):
        return embedding_function.embed_query(chunk.page_content)

    def track_progress(futures, total_tasks):
        with tqdm(total=total_tasks, desc="Progresso", unit="task") as pbar:
            for future in futures:
                future.result()  # Espera a conclusão da tarefa
                pbar.update(1)  # Atualiza a barra de progresso

    print("Gerando embeddings...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_embedding, chunk)
                   for chunk in chunks]
        track_progress(futures, len(chunks))
        embeddings = [future.result() for future in futures]

    print("Embeddings geradas. Salvando na base de dados...")

    # Add documents and embeddings to Chroma
    for idx, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk.page_content],
            metadatas=[
                {'id': idx, 'source': chunk.metadata.get('source', 'unknown')}],
            embeddings=[embeddings[idx]],
            ids=[str(idx)]
        )
    print("Base de dados atualizada com sucesso!")

# Initialize retriever using Ollama embeddings for queries
embedding_function = OllamaEmbeddings(model=MODELO_LLM)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    client=PersistentClient(path=CHROMADB_DIR),
    embedding_function=embedding_function
)
retriever = vectorstore.as_retriever(
    search_type=SEARCH_TYPE,
    search_kwargs={
        "k": NUM_DOCUMENTS,
        "fetch_k": FETCH_K,
        "lambda_mult": LAMBDA_MULT,
        "filter": None  # Pode ser usado para filtrar por metadados específicos
    }
)


def retrieve_context(question):
    results = retriever.invoke(question)

    # Adiciona informação sobre a fonte de cada trecho
    processed_results = []
    for doc in results:
        source = doc.metadata.get('source', 'fonte desconhecida')
        if isinstance(source, str) and source.endswith('.pdf'):
            source = os.path.basename(source)

        processed_text = f"[Fonte: {source}]\n{doc.page_content}"
        processed_results.append(processed_text)

    # Junta todos os trechos com separação clara
    context = "\n\n---\n\n".join(processed_results)
    return context


def is_portuguese(text):
    try:
        return detect(text) == 'pt'  # Verifica se o texto está em português
    except:
        return False


def query_llama(question, context):
    formatted_prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Contexto:\n{context}\n\n"
        f"Consulta: {question}\n\n"
        f"Resposta:"
    )
    response = ollama.generate(
        model=MODELO_LLM,
        prompt=formatted_prompt,
        options={"temperature": LLM_TEMPERATURE}
    )
    response_content = response['response'].strip()

    # Verifica se a resposta está em português
    if not is_portuguese(response_content):
        return "Desculpe, não consegui gerar uma resposta em português."

    return response_content


def ask_question(question):
    context = retrieve_context(question)
    answer = query_llama(question, context)
    return answer


# Set up the Gradio interface
interface = gr.Interface(
    fn=ask_question,
    inputs="text",
    outputs="text",
    title=UI_TITLE,
    description=UI_DESCRIPTION,
    allow_flagging="never",
    examples=EXAMPLES
)

# Launch the interface
interface.launch()
