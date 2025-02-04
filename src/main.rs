use futures::stream::Stream;
use langchain_rust::chain::{
    Chain, ConversationalRetrieverChain, ConversationalRetrieverChainBuilder,
};
use langchain_rust::embedding::ollama::ollama_embedder::OllamaEmbedder;
// use langchain_rust::language_models::llm::LLM;
// use langchain_rust::language_models::LLMError;
use langchain_rust::llm::ollama::client::Ollama;
use langchain_rust::memory::SimpleMemory;
use langchain_rust::prompt::{HumanMessagePromptTemplate, MessageFormatterStruct};
use langchain_rust::schemas::messages::Message;
use langchain_rust::schemas::{Document, StreamData};
use langchain_rust::vectorstore::qdrant::{Qdrant, Store, StoreBuilder};
use langchain_rust::{fmt_message, fmt_template, message_formatter, prompt_args, template_jinja2};
use std::io::Write;
use std::pin::Pin;
use std::vec;
use tokio_stream::StreamExt;

use langchain_rust::vectorstore::{Retriever, VecStoreOptions, VectorStore};

const LLM_MODEL: &str = "deepseek-r1:1.5b";
// const LLM_MODEL: &str = "llama3.2:1b";

const EMBEDDING_MODEL: &str = "nomic-embed-text";
const CONNECTION_URL: &str = "http://localhost:6334";

// 1536 dimensions, not 768

async fn add_document(store: &Store, documents: &Vec<Document>) {
    // let _ = add_documents!(store, documents).await.map_err(|e| {
    //     eprintln!("Error adding documents: {:?}", e);
    // });
    let vectorstore_options = VecStoreOptions::default();
    store
        .add_documents(&documents, &vectorstore_options)
        .await
        .unwrap();
}

async fn create_documetns() -> Vec<Document> {
    let documents = vec![
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "Which is the favorite text editor of luis", "Nvim"
        )),
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "How old is Luis", "24"
        )),
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "Where do luis live", "Peru"
        )),
        Document::new(format!(
            "\nQuestion: {}\nAnswer: {}\n",
            "Whats his favorite food", "Pan con chicharron"
        )),
    ];
    documents
}

async fn get_embedder() -> OllamaEmbedder {
    OllamaEmbedder::default().with_model(EMBEDDING_MODEL)
}

async fn base_init(collection_name: &String) {
    let store = get_store(collection_name).await;
    let documents: Vec<Document> = create_documetns().await;
    add_document(&store, &documents).await;
}

async fn get_store(collection_name: &String) -> Store {
    let embedding_model = get_embedder().await;
    let client: Qdrant = Qdrant::from_url(CONNECTION_URL).build().unwrap();

    StoreBuilder::new()
        .embedder(embedding_model)
        .client(client)
        .collection_name(collection_name)
        .build()
        .await
        .unwrap()
}

fn get_query() -> String {
    let mut query: String = String::new();
    std::io::stdout().flush().unwrap();
    std::io::stdin().read_line(&mut query).unwrap();
    query
}

fn get_prompt() -> MessageFormatterStruct {
    message_formatter![
        fmt_message!(Message::new_system_message("You are a helpful assistant. 
while answering you don't need to explictly mention the context, just answer the question in a natural way. If you are asked of the same question please answer that polietly.
Never makeup the answer.But you can always have a nice conversation with the user."
        )),
        
        fmt_template!(HumanMessagePromptTemplate::new(template_jinja2!("
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
If there is no context but the question is general,answer that nicely.
{{context}}

Question:{{question}}
Helpful Answer:
",
"context","question")))
    ]
}

async fn get_chain(collection_name: &String) -> ConversationalRetrieverChain {
    let store: Store = get_store(&collection_name).await;
    let prompt = get_prompt();

    let model: Ollama = Ollama::default().with_model(LLM_MODEL);

    ConversationalRetrieverChainBuilder::new()
        .llm(model)
        .rephrase_question(true)
        .memory(SimpleMemory::new().into())
        .retriever(Retriever::new(store, 5))
        .prompt(prompt)
        .build()
        .expect("error in chain")
}

async fn print_stream(
    stream: &mut Pin<
        Box<dyn Stream<Item = Result<StreamData, langchain_rust::chain::ChainError>> + Send>,
    >,
) {
    while let Some(res) = stream.next().await {
        match res {
            Ok(data) => {
                data.to_stdout().unwrap();
            }
            Err(e) => {
                eprintln!("Error: {:?}", e);
            }
        }
    }
    println!();
}

fn print_user_prompt() {
    print!("User> ");
    std::io::stdout().flush().unwrap();
}

fn print_ai_promt() {
    print!("AI> ");
    std::io::stdout().flush().unwrap();
}

async fn cli_chat(collection_name: &String) {
    let chain = get_chain(collection_name).await;
    let mut query: String;

    loop {
        print_user_prompt();
        query = get_query();
        if query.trim() == String::from("exit") {
            break;
        }

        if query.trim().is_empty() {
            continue;
        }

        print_ai_promt();
        let input_variables = prompt_args! {
            "question" => query
        };

        let mut stream: Pin<
            Box<dyn Stream<Item = Result<StreamData, langchain_rust::chain::ChainError>> + Send>,
        > = chain.stream(input_variables).await.unwrap();

        print_stream(&mut stream).await;
    }
}

#[tokio::main]
async fn main() {
    // let model_name: String = String::from("deepseek-r1:1.5b");
    println!("=============starting ollama ==================");
    let base_init_required: bool = false;
    let collection_name: String = String::from("rag_test_store");

    if base_init_required {
        base_init(&collection_name).await;
    }

    cli_chat(&collection_name).await;
}
