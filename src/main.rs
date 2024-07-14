mod embeddings;
mod loaders;

#[tokio::main]
async fn main() {
    let pdf_path = String::from("./src/data/lora.pdf");

    let loaded_docs = loaders::loaders::pdf_loader(&pdf_path).await;
    println!("Loaded docs: {:?}", loaded_docs.get(1));
    let query: String = String::from("what is lora?");
    let query_embedding = embeddings::embeddings::get_embedding(&query).await;
    println!("{:?}", query_embedding.get(1));
}
