from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format=os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
    filename=os.getenv('LOG_FILE', 'vector_search.log')
)
logger = logging.getLogger(__name__)

class VectorConverter:
    """
    Converts text data from Elasticsearch index to vector representations.
    """
    
    def __init__(
        self,
        es_host: str = os.getenv('ES_HOST', 'localhost'),
        es_port: int = int(os.getenv('ES_PORT', 9200)),
        original_index: str = os.getenv('ORIGINAL_INDEX', 'search-new-job-title-v2'),
        vector_index: str = os.getenv('VECTOR_INDEX', 'job_titles_vectors'),
        model_name: str = os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2'),
        batch_size: int = int(os.getenv('BATCH_SIZE', 100))
    ):
        """
        Initialize the converter with configuration parameters.
        """
        # Initialize Elasticsearch client
        self.es = Elasticsearch(
           es_host,
           es_port,
           api_key=os.getenv('ES_API_KEY'),
        )
        
        self.original_index = original_index
        self.vector_index = vector_index
        self.batch_size = batch_size
        
        # Initialize the sentence transformer model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Verify Elasticsearch connection
        if not self.es.ping():
            raise ConnectionError("Could not connect to Elasticsearch")
    
    def create_vector_index(self) -> None:
        """Create separate vector index"""
        try:
            vector_mapping = {
                "mappings": {
                    "properties": {
                        "original_id": {  # Reference to original document
                            "type": "keyword"
                        },
                        "job_title": {
                            "type": "keyword"
                        },
                        "title_vector": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "synonym_vector": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "combined_vector": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": int(os.getenv('NUMBER_OF_SHARDS', 2)),
                    "number_of_replicas": int(os.getenv('NUMBER_OF_REPLICAS', 1))
                }
            }
            
            self.es.indices.create(index=self.vector_index, body=vector_mapping)
            logger.info(f"Created vector index: {self.vector_index}")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            raise
    
    def encode_text(self, text: str) -> np.ndarray:
        """Convert text to vector embedding"""
        return self.model.encode(text)
    
    def calculate_combined_vector(
        self,
        title_vector: np.ndarray,
        synonym_vector: np.ndarray
    ) -> np.ndarray:
        """Calculate combined vector with weighted average"""
        title_weight = 0.6
        synonym_weight = 0.4
        combined = (title_vector * title_weight) + (synonym_vector * synonym_weight)
        return combined / np.linalg.norm(combined)
    
    def process_and_store_vectors(
        self,
        doc_id: str,
        job_title: str,
        synonyms: str
    ) -> None:
        """Process and store vectors for a single document"""
        try:
            # Generate vectors
            title_vector = self.encode_text(job_title)
            synonym_vector = self.encode_text(synonyms)
            combined_vector = self.calculate_combined_vector(title_vector, synonym_vector)
            
            # Prepare vector document
            vector_doc = {
                "original_id": doc_id,
                "job_title": job_title,
                "title_vector": title_vector.tolist(),
                "synonym_vector": synonym_vector.tolist(),
                "combined_vector": combined_vector.tolist()
            }
            
            # Index vector document
            self.es.index(index=self.vector_index, body=vector_doc)
            logger.info(f"Processed and stored vectors for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            raise
    
    def bulk_process_existing_data(
        self,
        batch_size: int = int(os.getenv('BATCH_SIZE', 100))
    ) -> None:
        """Process existing data in batches"""
        try:
            query = {
                "query": {"match_all": {}},
                "size": batch_size
            }
            
            # Scroll through original index
            scroll = self.es.search(
                index=self.original_index,
                body=query,
                scroll='5m'
            )
            scroll_id = scroll['_scroll_id']
            
            total_processed = 0
            
            while True:
                batch_docs = scroll['hits']['hits']
                if not batch_docs:
                    break
                
                vector_actions = []
                for doc in batch_docs:
                    try:
                        source = doc['_source']
                        
                        # Generate vectors
                        title_vector = self.encode_text(source['job_title'])
                        synonym_vector = self.encode_text(source['synonyms'])
                        combined_vector = self.calculate_combined_vector(title_vector, synonym_vector)
                        
                        # Prepare vector document
                        vector_doc = {
                            "_index": self.vector_index,
                            "_source": {
                                "original_id": doc['_id'],
                                "job_title": source['job_title'],
                                "title_vector": title_vector.tolist(),
                                "synonym_vector": synonym_vector.tolist(),
                                "combined_vector": combined_vector.tolist()
                            }
                        }
                        vector_actions.append(vector_doc)
                        
                    except Exception as e:
                        logger.error(f"Error processing document {doc.get('_id')}: {str(e)}")
                        continue
                
                # Bulk index vector documents
                if vector_actions:
                    helpers.bulk(self.es, vector_actions)
                    total_processed += len(vector_actions)
                    logger.info(f"Processed {total_processed} documents")
                
                # Get next batch
                scroll = self.es.scroll(scroll_id=scroll_id, scroll='5m')
            
            logger.info(f"Completed processing {total_processed} documents")
            
        except Exception as e:
            logger.error(f"Error in bulk processing: {str(e)}")
            raise
        finally:
            try:
                # Clear scroll context
                self.es.clear_scroll(scroll_id=scroll_id)
            except:
                pass

if __name__ == "__main__":
    try:
        # Initialize the system
        job_system = VectorConverter()
        
        # Create vector index (only needed once)
        try:
            job_system.create_vector_index()
            print("Vector index created successfully")
        except Exception as e:
            print(f"Index creation error (might already exist): {e}")
        
        # Process existing data
        try:
            job_system.bulk_process_existing_data()
            print("Bulk processing completed")
        except Exception as e:
            print(f"Bulk processing error: {e}")
            
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise