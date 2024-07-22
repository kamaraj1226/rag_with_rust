pub mod embeddings {
    use langchain_rust::embedding::{Embedder, OllamaEmbedder};

    pub async fn get_embedding(query: &String, embedding_model: &String) -> Vec<f64> {
        let ollama_embedder = get_embedder(embedding_model);
        let response = ollama_embedder.embed_query(query).await.unwrap();

        response
    }

    pub fn get_embedder(embedding_model: &String) -> OllamaEmbedder {
        let ollama_embedder = OllamaEmbedder::default().with_model(embedding_model);
        ollama_embedder
    }
}
