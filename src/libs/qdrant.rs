pub mod qdrant {
    use crate::libs::embeddings::embeddings::Embeddings;
    use crate::libs::loaders::loaders::pdf_loader;
    use langchain_rust::vectorstore::{
        qdrant::{Qdrant, Store, StoreBuilder},
        VecStoreOptions, VectorStore,
    };

    pub struct QdrantBase {
        pub emdedder: Embeddings,
        pub collection_name: String,
        pub qdrant_url: String,
    }

    impl QdrantBase {
        fn get_client(&self) -> Qdrant {
            let qdrant_url = &self.qdrant_url;
            let client = Qdrant::from_url(&qdrant_url).build().unwrap();
            client
        }

        async fn check_collection_exists(&self, collection_name: &String) -> bool {
            let client: Qdrant = self.get_client();
            let exist = client.collection_exists(collection_name).await;

            match exist {
                Ok(res) => return res,
                Err(_e) => return false,
            }
        }

        pub async fn create_collection(&self) -> Store {
            let client = self.get_client();
            let collection_name = &self.collection_name;
            let embedder = self.emdedder.get_embedder();
            let store: Store = StoreBuilder::new()
                .embedder(embedder)
                .client(client)
                .collection_name(&collection_name)
                .build()
                .await
                .unwrap();
            store
        }

        pub async fn add_pdf(&self, pdf_path: String, force: bool) {
            let loaded_docs = pdf_loader(&pdf_path).await;

            if !self.check_collection_exists(&self.collection_name).await {
                println!(
                    "creating new collection collection_name: {}",
                    &self.collection_name
                );
                let store = self.create_collection().await;
                store
                    .add_documents(&loaded_docs, &VecStoreOptions::default())
                    .await
                    .unwrap();
            } else {
                if !force {
                    println!("Collection already exists Skipping adding file");
                    return;
                }
                println!(
                    "Adding docs to existing collection collection_name: {}",
                    &self.collection_name
                );
                let client = self.get_client();
                let store_builder = StoreBuilder::new();
                let store = store_builder
                    .embedder(self.emdedder.get_embedder())
                    .client(client)
                    .collection_name(&self.collection_name)
                    .build()
                    .await
                    .unwrap();
                store
                    .add_documents(&loaded_docs, &VecStoreOptions::default())
                    .await
                    .unwrap();
            }
        }
    }
}
