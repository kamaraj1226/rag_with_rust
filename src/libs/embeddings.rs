pub mod embeddings {

    use langchain_rust::embedding::{Embedder, OllamaEmbedder};

    #[derive(Clone)]
    pub struct Embeddings {
        pub embedding_model: String,
    }

    impl Embeddings {
        pub async fn get_embedding(&self, query: &String) -> Vec<f64> {
            let ollama_embedder = self.get_embedder();
            let response = ollama_embedder.embed_query(query).await.unwrap();

            response
        }

        pub fn get_embedder(&self) -> OllamaEmbedder {
            let ollama_embedder = OllamaEmbedder::default().with_model(&self.embedding_model);
            ollama_embedder
        }
    }
}
