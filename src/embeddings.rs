pub mod embeddings {
    use langchain_rust::embedding::{Embedder, OllamaEmbedder};

    pub async fn get_embedding(query: &String) -> Vec<f64> {
        let model_name = String::from("mxbai-embed-large");
        let ollama = OllamaEmbedder::default().with_model(model_name);

        let response = ollama.embed_query(query).await.unwrap();

        response
    }
}
