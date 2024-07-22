mod embeddings;
mod loaders;
mod qdrant;

#[tokio::main]
async fn main() {
    // let pdf_path = String::from("./src/data/lora.pdf");

    // let loaded_docs = loaders::loaders::pdf_loader(&pdf_path).await;
    // println!("Loaded docs: {:?}", loaded_docs.get(1));
    let embedding_model = String::from("mxbai-embed-large");
    let collection_name = String::from("lora");
    let query: String = String::from("what is lora?");
    let query_embedding = embeddings::embeddings::get_embedding(&query, &embedding_model).await;
    println!("{:?}", query_embedding.get(1));

    let embedder = embeddings::embeddings::get_embedder(&embedding_model);
    qdrant::qdrant::create_collection(embedder, &collection_name).await;
}
