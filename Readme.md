# elasticsearch-semantic-jobs

Convert job titles and their synonyms into vector embeddings for semantic search using Elasticsearch. This project provides tools to transform existing Elasticsearch indices into vector representations and perform semantic searches using these vectors.

## 🚀 Features

- Convert existing Elasticsearch job title indices to vector representations
- Multiple vector search implementations
  - Basic vector search
  - Hybrid search (combining text and vector)
  - Multi-vector weighted search
- Built-in performance monitoring and logging
- Batch processing with progress tracking
- Configurable model selection
- Support for synonyms and combined vectors
- Comprehensive error handling

## 📋 Prerequisites

- Python 3.8+
- Elasticsearch 8.x
- Existing Elasticsearch index with job titles and synonyms
- At least 4GB RAM (for model loading)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/parthrupapara/elasticsearch-semantic-jobs.git
cd elasticsearch-semantic-jobs
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

## ⚙️ Configuration

Example `.env` configuration:
```env
# Elasticsearch Configuration
ES_HOST=localhost
ES_PORT=9200
ES_USERNAME=elastic
ES_PASSWORD=your_password
ES_USE_SSL=true
ES_API_KEY=

# Index Configuration
ORIGINAL_INDEX=search-new-job-title
VECTOR_INDEX=job_titles_vectors

# Model Configuration
MODEL_NAME=all-MiniLM-L6-v2
BATCH_SIZE=100
```

## 🎯 Usage

### Converting Existing Index to Vectors

```python
from src.vector_search.converter import VectorConverter

# Initialize converter
converter = VectorConverter()

# Create vector index (only needed once)
converter.create_vector_index()

# Process all documents
converter.bulk_process_existing_data()

# Get processing statistics
stats = converter.get_processing_stats()
print(f"Processing statistics: {stats}")
```

### Performing Vector Search

```python
from src.vector_search.search import VectorSearch

# Initialize search
search = VectorSearch()

# Basic vector search
results = search.basic_vector_search(
    query_text="senior software engineer",
    top_k=10,
    min_score=0.5
)

# Print results
for hit in results:
    print(f"Score: {hit['_score']:.3f} - Title: {hit['_source']['job_title']}")
```

## 📊 Performance Monitoring

The system includes built-in performance monitoring:

```python
Query Start Time: 2024-11-23 10:30:15.123456
Vector Encoding Time: 0.123 seconds
Query Duration: 0.234 seconds
Total Results Found: 10
```

## 📝 Code Examples

### Vector Conversion Example

```python
# Initialize converter with custom settings
converter = VectorConverter(
    es_host="localhost",
    es_port=9200,
    original_index="my-job-titles",
    vector_index="job-vectors"
)

# Process a single document
doc_vectors = converter.process_single_document("doc_id_123")
```

### Search Example

```python
# Initialize search with custom settings
search = VectorSearch(
    es_host="localhost",
    es_port=9200,
    vector_index="job-vectors"
)

# Perform hybrid search
results = search.hybrid_search(
    query_text="software architect",
    text_boost=0.3,
    vector_boost=0.7,
    top_k=5
)
```

## 🔍 Search Types

1. Basic Vector Search
   - Pure vector similarity search
   - Uses cosine similarity
   - Best for semantic matching

2. Hybrid Search
   - Combines text and vector search
   - Configurable weights
   - Best for balanced results

3. Multi-Vector Search
   - Uses title, synonym, and combined vectors
   - Weighted scoring
   - Best for comprehensive matching

## 🎯 Use Cases

1. Job Title Matching
   - Match job titles semantically
   - Handle variations and synonyms
   - Find related roles

2. Skill Matching
   - Match similar skills
   - Handle different terminology
   - Find related competencies

3. Resume Parsing
   - Extract and match job titles
   - Handle different formats
   - Find relevant experience

## ⚠️ Error Handling

The system includes comprehensive error handling:

```python
try:
    results = search.basic_vector_search("developer")
except ConnectionError:
    logger.error("Could not connect to Elasticsearch")
except Exception as e:
    logger.error(f"Search error: {str(e)}")
```

## 📈 Monitoring and Logging

Logs are saved in `vector_search.log`:
```
2024-11-23 10:30:15 INFO Starting vector conversion...
2024-11-23 10:30:15 INFO Vector encoding completed in 0.123s
2024-11-23 10:30:16 INFO Processed 1000/5000 documents
```

## 📊 Performance Tips

1. Batch Processing
   - Use appropriate batch sizes
   - Monitor memory usage
   - Use scroll API for large datasets

2. Search Optimization
   - Adjust min_score for better results
   - Fine-tune boost values
   - Use appropriate vector fields

3. Index Configuration
   - Optimize number of shards
   - Configure refresh interval
   - Set appropriate replicas

## 🔗 Links

- [Project Homepage](https://github.com/parthrupapara/elasticsearch-semantic-jobs)

