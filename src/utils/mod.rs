use crate::CONNECTION_URL;

// use langchain_rust::embedding::ollama::ollama_embedder::OllamaEmbedder;
use futures::stream::Stream;
use langchain_rust::llm::client::Ollama;
use langchain_rust::memory::SimpleMemory;
use langchain_rust::prompt::{HumanMessagePromptTemplate, MessageFormatterStruct};
use langchain_rust::schemas::messages::Message;
use langchain_rust::schemas::Document;
use langchain_rust::schemas::StreamData;
use langchain_rust::vectorstore::qdrant::{Qdrant, Store, StoreBuilder};
use langchain_rust::vectorstore::Retriever;
use langchain_rust::vectorstore::{VecStoreOptions, VectorStore};
use langchain_rust::{fmt_message, fmt_template, message_formatter, template_jinja2};
use std::io::Write;
use std::pin::Pin;
use std::sync::Arc;
use std::vec;
use tokio::sync::Mutex;
use tokio_stream::StreamExt;
pub mod chat;
pub mod embedder;
pub mod models;

pub async fn create_documetns() -> Vec<Document> {
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

async fn add_document(store: &Store, documents: &Vec<Document>) {
    let vectorstore_options = VecStoreOptions::default();
    store
        .add_documents(&documents, &vectorstore_options)
        .await
        .unwrap();
}

pub async fn base_init(collection_name: &str, embedding_model: &str) {
    let store = get_store(collection_name, embedding_model).await;
    let documents: Vec<Document> = create_documetns().await;
    add_document(&store, &documents).await;
}

pub async fn get_store(collection_name: &str, embedding_model: &str) -> Store {
    let embedding_model = embedder::get_embedder(embedding_model).await;
    let client: Qdrant = Qdrant::from_url(CONNECTION_URL).build().unwrap();

    StoreBuilder::new()
        .embedder(embedding_model)
        .client(client)
        .collection_name(collection_name)
        .build()
        .await
        .unwrap()
}

pub fn get_query() -> String {
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

pub fn get_memory() -> Arc<Mutex<SimpleMemory>> {
    Arc::new(Mutex::new(SimpleMemory::new()))
}

fn get_retriever(store: Store) -> Retriever {
    let num_docs = 2;
    Retriever::new(store, num_docs)
}

type StreamType =
    Pin<Box<dyn Stream<Item = Result<StreamData, langchain_rust::chain::ChainError>> + Send>>;

pub async fn print_stream(stream: &mut StreamType) {
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

pub fn print_user_prompt() {
    print!("User> ");
    std::io::stdout().flush().unwrap();
}

pub fn print_ai_promt() {
    print!("AI> ");
    std::io::stdout().flush().unwrap();
}

pub struct ChatChainArgs {
    pub prompt: MessageFormatterStruct,
    pub model: Ollama,
    pub memory: Arc<Mutex<SimpleMemory>>,
    pub retriever: Retriever,
}

pub async fn get_chain_args(
    collection_name: &str,
    embedding_model: &str,
    model: Ollama,
) -> ChatChainArgs {
    ChatChainArgs {
        prompt: get_prompt(),
        model,
        memory: get_memory(),
        retriever: get_retriever(get_store(&collection_name, embedding_model).await),
    }
}

pub async fn get_chain(args: ChatChainArgs) -> langchain_rust::chain::ConversationalRetrieverChain {
    let rephrase_question = true;

    langchain_rust::chain::ConversationalRetrieverChainBuilder::new()
        .llm(args.model.clone())
        .rephrase_question(rephrase_question)
        .memory(args.memory)
        .retriever(args.retriever)
        .prompt(args.prompt)
        .build()
        .expect("error in chain")
}

pub async fn get_model(model_name: &str) -> Ollama {
    Ollama::default().with_model(model_name)
}
