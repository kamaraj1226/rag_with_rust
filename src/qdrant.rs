pub mod qdrant {
    use langchain_rust::embedding::OllamaEmbedder;
    use langchain_rust::vectorstore::qdrant::{Qdrant, Store, StoreBuilder};

    pub fn get_client() -> Qdrant {
        let qdrant_url = String::from("http://localhost:6334");
        let client = Qdrant::from_url(&qdrant_url).build().unwrap();
        client
    }

    pub async fn create_collection(embedder: OllamaEmbedder, collection_name: &String) -> Store {
        let client = get_client();
        let store: Store = StoreBuilder::new()
            .embedder(embedder)
            .client(client)
            .collection_name(collection_name)
            .build()
            .await
            .unwrap();
        store
    }
}
