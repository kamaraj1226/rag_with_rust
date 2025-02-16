use crate::utils::{self, chat::Chat, models::LlmModels};
use langchain_rust::{
    chain::Chain, llm::client::Ollama, memory::SimpleMemory, prompt_args, schemas::BaseMemory,
};
use std::sync::Arc;
use tokio::sync::Mutex;

pub mod cli;

#[derive(Clone)]
pub struct CliChat<'a, 'b> {
    embedding_model_name: &'a str,
    collection_name: &'b str,
    pub model: Ollama,
    memory: Arc<Mutex<SimpleMemory>>,
}

impl<'a, 'b> CliChat<'a, 'b> {
    pub async fn new(
        embedding_model_name: &'a str,
        model_name: LlmModels,
        collection_name: &'b str,
    ) -> Self {
        let model = utils::get_model(model_name.as_str()).await;
        let memory = utils::get_memory();

        CliChat {
            embedding_model_name,
            collection_name,
            model,
            memory,
        }
    }

    pub async fn cli_chat(&self) {
        let memory = self.memory.clone();
        let chain = self.get_chain().await;
        // chain.memory
        let mut query: String;

        loop {
            utils::print_user_prompt();
            query = utils::get_query();
            if query.trim() == String::from("exit") {
                break;
            }

            if query.trim() == String::from("clear") {
                memory.lock().await.clear();
                println!("Memory cleared");
                continue;
            }

            if query.trim().is_empty() {
                continue;
            }

            utils::print_ai_promt();
            let input_variables = prompt_args! {
                "question" => query
            };

            let mut stream: utils::chat::StreamType = chain.stream(input_variables).await.unwrap();

            self.print_stream(&mut stream).await;
        }
    }
}
