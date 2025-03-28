# rag-chatbot

## Descrição

O **rag-chatbot** é um assistente virtual desenvolvido para responder a consultas sobre documentos normativos do Instituto Federal do Piauí (IFPI). Utiliza técnicas de recuperação de informações e embeddings para fornecer respostas precisas e contextuais.

## Funcionalidades

- Respostas baseadas em documentos normativos institucionais.
- Geração de embeddings para melhorar a recuperação de informações.
- Interface amigável utilizando Gradio.

## Estrutura do Projeto

- `chatbot.py`: Código principal do chatbot, incluindo a lógica de recuperação e geração de respostas.
- `docs/`: Diretório contendo os documentos PDF a serem consultados.

## Como Usar

1. Certifique-se de que a pasta `docs` contém os arquivos PDF necessários.
2. Execute o script `chatbot.py`.
3. Acesse a interface do Gradio para fazer perguntas.

## Dependências

- `ollama`
- `gradio`
- `langchain`
- `chromadb`
- `langdetect`
- `tqdm`

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## Licença

Este projeto está licenciado sob a MIT License.
