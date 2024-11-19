# CISI Dataset for Information Retrieval

This repository contains the CISI dataset, a collection of documents, queries, and ground truth relevance judgments used for information retrieval tasks.

## Dataset Description

- **documents.csv**: Contains 1,460 documents, each with a unique ID, title, author, and abstract.
- **queries.csv**: Contains 112 queries, each with a unique ID and query text.
- **ground_truth.csv**: Contains the mapping of query ID to document ID, representing the "gold standard" or "ground truth" of query-document matching.

## Format

Each file is in CSV format with the following columns:

### documents.csv
- `doc_id`: Unique identifier for each document.
- `title`: Title of the document.
- `author`: Author of the document.
- `text`: Abstract or main content of the document.

### queries.csv
- `query_id`: Unique identifier for each query.
- `text`: Text of the query.

### ground_truth.csv
- `query_id`: Identifier of the query.
- `doc_id`: Identifier of the document relevant to the query.

## Usage

This dataset can be used for training and evaluating information retrieval models. The ground truth data provides a benchmark for comparing the performance of different retrieval algorithms.

## Acknowledgment

This dataset has been made publicly available by the Information Retrieval Group at the University of Glasgow. We express our gratitude for their contribution to the research community.

## License

This dataset is released under the [MIT License](LICENSE).
