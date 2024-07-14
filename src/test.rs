use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_template,
    llm::ollama,
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::messages::Message,
    template_fstring,
};

fn get_chain() {
    // let ollama = ollama::client::Ollama::default().with_model("llama3");
    // let response = ollama.invoke("Hi").await.unwrap();
    // println!("{}", response);

    // let prompt = message_formatter![
    //     fmt_message!(Message::new_system_message(
    //         "You are a good asistant. you are an expert in mathematics."
    //     )),
    //     fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
    //         "{input}", "input"
    //     )))
    // ];
    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(ollama.clone())
        .build()
        .unwrap();

    match chain
        .invoke(prompt_args! {
            "input" => "can you solve 2+2"
        })
        .await
    {
        Ok(result) => {
            println!("Result: {:?}", result);
        }
        Err(e) => println!("Error invoking LLMChain: {:?}", e),
    }
}
