# End-To-End Face Recognition System with ZenML and FastAPI

## Visão Geral do Projeto

### Objetivo
Desenvolver uma aplicação de reconhecimento facial que permita verificar a similaridade entre uma imagem de entrada e imagens em uma base de dados, com a possibilidade de adicionar novas faces à base.

### Tecnologias
- **Python**
- **FastAPI**
- **Docker**
- **ZenML**
- **Redes Siamesas**
- **Banco de dados** (SQLite/PostgreSQL)
- **FAISS** (opcional)

## Requisitos Funcionais

1. **Treinamento do modelo de rede siamesa** com dados faciais para gerar embeddings que representam cada face.
2. **Implementar um pipeline de treinamento modular com ZenML**, incluindo steps para pré-processamento, treinamento, e avaliação.
3. **Criar uma API REST com FastAPI para**:
   - 3.1: Receber uma nova imagem, extrair as características e comparar com as existentes no banco para retornar a mais semelhante.
   - 3.2: Permitir adicionar uma nova face à base, armazenando o embedding correspondente.
4. **Configurar um banco de dados** para armazenar as características faciais (embeddings) e informações das faces registradas.
5. **Implantar a API e o banco de dados** como serviços Docker para garantir portabilidade e escalabilidade.
