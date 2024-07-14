pub mod loaders {
    use futures_util::StreamExt;
    use langchain_rust::document_loaders::{lo_loader::LoPdfLoader, Loader};
    use langchain_rust::schemas::Document;

    pub async fn pdf_loader(pdf_path: &String) -> Vec<Document> {
        let loader = LoPdfLoader::from_path(pdf_path)
            .expect(format!("Can't able to load pdf. Check the path: PATH: {}", pdf_path).as_str());
        let docs = loader
            .load()
            .await
            .unwrap()
            .map(|d| d.unwrap())
            .collect::<Vec<_>>()
            .await;

        docs
    }
}
