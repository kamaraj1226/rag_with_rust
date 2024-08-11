mod libs {
    mod embeddings;
    mod loaders;
    mod qdrant;
    pub use embeddings::embeddings::Embeddings;
    pub use qdrant::qdrant::QdrantBase;
}

use libs::Embeddings;
use libs::QdrantBase;

#[tokio::main]
async fn main() {
    let embedding_model = String::from("mxbai-embed-large");
    let collection_name = String::from("lora");
    let qdrant_url = String::from("http://localhost:6334");

    let _embedding = Embeddings { embedding_model };

    let _qdrant = QdrantBase {
        emdedder: _embedding.clone(),
        collection_name,
        qdrant_url,
    };

    const TOLOAD: bool = false;
    if TOLOAD {
        let pdf_path = String::from("./src/data/lora.pdf");
        _qdrant.add_pdf(pdf_path, false).await;
    }

    let query: String = String::from("what is lora?");
    let query_embedding = _embedding.get_embedding(&query).await;
    println!("{:?}", query_embedding.get(1));
}
