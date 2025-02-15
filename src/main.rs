use futures::stream::Stream;
use langchain_rust::chain::{
    Chain, ConversationalRetrieverChain, ConversationalRetrieverChainBuilder,
};
use langchain_rust::embedding::ollama::ollama_embedder::OllamaEmbedder;
use langchain_rust::llm::ollama::client::Ollama;
use langchain_rust::memory::SimpleMemory;
use langchain_rust::prompt::{HumanMessagePromptTemplate, MessageFormatterStruct};
use langchain_rust::schemas::messages::Message;
use langchain_rust::schemas::{BaseMemory, Document, StreamData};
use langchain_rust::vectorstore::qdrant::{Qdrant, Store, StoreBuilder};
use langchain_rust::{fmt_message, fmt_template, message_formatter, prompt_args, template_jinja2};
use tokio::sync::Mutex;

use std::io::Write;
use std::pin::Pin;
use std::sync::Arc;
use std::vec;
use tokio_stream::StreamExt;

use langchain_rust::vectorstore::{Retriever, VecStoreOptions, VectorStore};

// const LLM_MODEL: &str = "deepseek-r1:1.5b";
const LLM_MODEL: &str = "llama3.2:1b";

const EMBEDDING_MODEL: &str = "nomic-embed-text";
const CONNECTION_URL: &str = "http://localhost:6334";

async fn add_document(store: &Store, documents: &Vec<Document>) {
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

fn get_memory() -> Arc<Mutex<SimpleMemory>> {
    Arc::new(Mutex::new(SimpleMemory::new()))
}

fn get_retriever(store: Store) -> Retriever {
    let num_docs = 5;
    Retriever::new(store, num_docs)
}

type StreamType =
    Pin<Box<dyn Stream<Item = Result<StreamData, langchain_rust::chain::ChainError>> + Send>>;

async fn print_stream(stream: &mut StreamType) {
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

async fn get_chain(args: ChatChainArgs) -> ConversationalRetrieverChain {
    let model = args.model;
    let prompt = args.prompt;
    let memory = args.memory;
    let retriever = args.retriever;

    ConversationalRetrieverChainBuilder::new()
        .llm(model)
        .rephrase_question(true)
        .memory(memory)
        .retriever(retriever)
        .prompt(prompt)
        .build()
        .expect("error in chain")
}

async fn cli_chat(args: ChatChainArgs) {
    let memory = args.memory.clone();
    let chain = get_chain(args).await;
    // chain.memory
    let mut query: String;

    loop {
        print_user_prompt();
        query = get_query();
        println!("query: {}", query);
        if query.trim() == String::from("exit") {
            break;
        }

        if query.trim() == String::from("clear") {
            memory.lock().await.clear();
        }

        if query.trim().is_empty() {
            continue;
        }

        print_ai_promt();
        let input_variables = prompt_args! {
            "question" => query
        };

        let mut stream: StreamType = chain.stream(input_variables).await.unwrap();

        print_stream(&mut stream).await;
    }
}

struct ChatChainArgs {
    prompt: MessageFormatterStruct,
    model: Ollama,
    memory: Arc<Mutex<SimpleMemory>>,
    retriever: Retriever,
}

#[tokio::main]
async fn main() {
    // let model_name: String = String::from("deepseek-r1:1.5b");
    println!("=============starting ollama ==================");
    let model: Ollama = Ollama::default().with_model(LLM_MODEL);

    let base_init_required: bool = false;
    let collection_name: String = String::from("rag_test_store");

    if base_init_required {
        base_init(&collection_name).await;
    }

    let args = ChatChainArgs {
        prompt: get_prompt(),
        model,
        memory: get_memory(),
        retriever: get_retriever(get_store(&collection_name).await),
    };
    cli_chat(args).await;
}
