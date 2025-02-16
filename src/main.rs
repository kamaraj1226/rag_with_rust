mod cli_chat;
mod utils;

const EMBEDDING_MODEL: &str = "nomic-embed-text";
const CONNECTION_URL: &str = "http://localhost:6334";

#[tokio::main]
async fn main() {
    // let model_name: String = String::from("deepseek-r1:1.5b");
    println!("=============starting ollama ==================");
    let base_init_required: bool = false;
    let collection_name: &str = "rag_test_store";

    if base_init_required {
        utils::base_init(collection_name, &EMBEDDING_MODEL).await;
    }

    let cli_chat = cli_chat::CliChat::new(
        &EMBEDDING_MODEL,
        utils::models::LlmModels::DeepseekR1_8b,
        collection_name,
    )
    .await;

    cli_chat.cli_chat().await;
}
