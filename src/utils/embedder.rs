use langchain_rust::embedding::ollama::ollama_embedder::OllamaEmbedder;

pub async fn get_embedder(embedding_model: &str) -> OllamaEmbedder {
    OllamaEmbedder::default().with_model(embedding_model)
}
