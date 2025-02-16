use futures::stream::Stream;
use langchain_rust::chain::ConversationalRetrieverChain;

use langchain_rust::schemas::StreamData;
use std::pin::Pin;

pub type StreamType =
    Pin<Box<dyn Stream<Item = Result<StreamData, langchain_rust::chain::ChainError>> + Send>>;

pub trait Chat {
    async fn get_chain(&self) -> ConversationalRetrieverChain;
    async fn print_stream(&self, stream: &mut StreamType);
}
