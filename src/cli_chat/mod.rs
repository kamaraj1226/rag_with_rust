use crate::utils::{self, chat::Chat, models::LlmModels};
use langchain_rust::{chain::Chain, llm::client::Ollama, prompt_args};

pub mod cli;

#[derive(Clone)]
pub struct CliChat<'a, 'b> {
    embedding_model_name: &'a str,
    collection_name: &'b str,
    model: Ollama,
}

impl<'a, 'b> CliChat<'a, 'b> {
    pub async fn new(
        embedding_model_name: &'a str,
        model_name: LlmModels,
        collection_name: &'b str,
    ) -> Self {
        let model = utils::get_model(model_name.as_str()).await;

        CliChat {
            embedding_model_name,
            collection_name,
            model,
        }
    }

    pub async fn cli_chat(&self) {
        // let memory = self.memory;
        let chain = self.get_chain().await;
        // chain.memory
        let mut query: String;

        loop {
            utils::print_user_prompt();
            query = utils::get_query();
            if query.trim() == String::from("\\exit") {
                break;
            }

            if query.trim() == String::from("\\clear") {
                chain.memory.lock().await.clear();
                println!("Memory cleared");
                continue;
            }

            if query.trim() == String::from("\\show") {
                let memory = chain.memory.lock().await.messages();
                println!("{:?}", memory);
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
