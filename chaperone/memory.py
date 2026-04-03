from langchain_community.document_loaders import WebBaseLoader

class RAGMemory:
    def __init__(self):
        print("\nFetching Official RFdiffusion Documentation...")
        loader = WebBaseLoader(
            web_paths=("https://raw.githubusercontent.com/RosettaCommons/RFdiffusion/main/README.md",),
        )
        docs = loader.load()
        self.rfdiffusion_docs = docs[0].page_content[:3000]

    def get_docs(self):
        return self.rfdiffusion_docs
