use super::CliChat;
use crate::utils;
use crate::utils::chat::{Chat, StreamType};

impl<'a, 'b> Chat for CliChat<'a, 'b> {
    async fn get_chain(&self) -> langchain_rust::chain::ConversationalRetrieverChain {
        let chain_args = utils::get_chain_args(
            self.collection_name,
            self.embedding_model_name,
            self.model.clone(),
        )
        .await;

        utils::get_chain(chain_args).await
    }

    async fn print_stream(&self, stream: &mut StreamType) {
        utils::print_stream(stream).await;
    }
}
