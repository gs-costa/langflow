from typing import Any

from astrapy.db import AstraDB
from langchain_core.documents import Document

from langflow.components.vectorstores.astradb import AstraDBVectorStoreComponent
from langflow.io import Output, TabInput
from langflow.schema.data import Data


class AstraDBDeleteComponent(AstraDBVectorStoreComponent):
    display_name = "Astra DB Delete Records"
    description = "Delete records from an Astra DB collection based on a filter query"
    documentation: str = "https://docs.datastax.com/en/astra-db-serverless/api-reference/row-methods/delete-many.html"
    icon = "AstraDB"
    name = "AstraDBDeleteRecords"

    inputs = [
        TabInput(
            name="mode",
            display_name="Mode",
            options=["Ingest", "Delete"],
            value="Ingest",
            info="Select the method to use",
            real_time_refresh=True,
        ),
        *AstraDBVectorStoreComponent.inputs,
    ]
    outputs = [
        Output(display_name="Search Results", name="search_results", method="search_documents"),
        Output(display_name="Deleted Records", name="deleted_records", method="delete_records"),
    ]

    client = None

    def initialize_client(self, token: str, api_endpoint: str) -> None:
        """Initialize the Astra DB client connection."""
        self.client = AstraDB(token=token, api_endpoint=api_endpoint)

    def delete_records(self) -> list[Data]:
        """Delete records from an Astra DB collection based on a filter query.

        Returns:
            dict[str, Any]: Result of the deletion operation including the number of deleted records
        """
        try:
            if not self.client:
                self.initialize_client(self.token, self.api_endpoint)

            if not self.client:
                raise Exception("Failed to initialize AstraDB client")

            collection = self.client.collection(self.collection_name)
            if not collection:
                raise Exception(f"Collection {self.collection_name} not found")

            documents = self.search_documents()

            result = collection.delete_many(self.__build_filter(documents))

            self.log(f"Deleted {result['status']['deletedCount']} records")

            return documents[:5]

        except Exception as e:
            error_message = f"Failed to delete records: {str(e)}"
            return self.__create_result_dict(deleted_count=0, success=False, message=error_message)

    def __create_result_dict(
        self, deleted_count: int, success: bool, message: str, dataframe: list[Data] | None = None
    ) -> list[Data]:
        return [
            Data(
                text_key="message",
                data={
                    "deleted_count": deleted_count,
                    "success": success,
                    "message": message,
                    "dataframe": dataframe,
                },
            )
        ]

    def search_documents(self, vector_store=None) -> list[Data]:
        docs = self._search_documents(vector_store)
        data = self.__docs_to_data(docs)

        self.log(f"Converted documents to data: {len(data)}")
        self.status = data

        return data

    def __docs_to_data(self, documents: list[Document]) -> list[Data]:
        return [self.__from_document(document) for document in documents]

    def __from_document(self, document: Document) -> Data:
        data = document.metadata
        data["text"] = document.page_content
        data["id"] = document.id
        return Data(data=data, text_key="text")

    def __build_filter(self, documents: list[Data]) -> dict[str, Any]:
        if self.advanced_search_filter:
            return self.advanced_search_filter
        ids = [doc.id for doc in documents]

        return {"_id": {"$in": ids}}
