from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from datetime import datetime
import time
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class JobTitleSearch:
    def __init__(
        self,
        es_host: str = os.getenv('ES_HOST', 'localhost'),
        es_port: int = int(os.getenv('ES_PORT', 9200))
    ):
        """Initialize the search system"""
        
        # Initialize Elasticsearch client
        self.es = Elasticsearch(
           es_host,
           es_port,
           api_key=os.getenv('ES_API_KEY'),
        )
        
        # Define index names
        self.original_index = os.getenv('ORIGINAL_INDEX', 'search-new-job-title-v2')
        self.vector_index = os.getenv('VECTOR_INDEX', 'job_titles_vectors')
        
        # Initialize the sentence transformer model
        self.model = SentenceTransformer(os.getenv('MODEL_NAME', 'all-MiniLM-L6-v2'))
        
        # Verify Elasticsearch connection
        if not self.es.ping():
            raise ConnectionError("Could not connect to Elasticsearch")

    def encode_text(self, text: str) -> np.ndarray:
        """Convert text to vector embedding"""
        try:
            start_time = time.time()
            vector = self.model.encode(text)
            encode_time = time.time() - start_time
            logger.debug(f"Text encoding took {encode_time:.3f} seconds")
            return vector
        except Exception as e:
            logger.error(f"Error encoding text: {str(e)}")
            raise

    def vector_search(
        self,
        query_text: str,
        top_k: int = 5,
        search_type: str = "combined",
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Search using vector similarity
        search_type options: "title", "synonym", "combined"
        """
        try:
            start_time = time.time()
            
            # Encode query
            query_vector = self.encode_text(query_text)
            
            # Select vector field based on search type
            vector_field = f"{search_type}_vector"
            
            # Prepare script query
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                        "params": {"query_vector": query_vector.tolist()}
                    }
                }
            }
            
            # Execute vector search
            vector_results = self.es.search(
                index=self.vector_index,
                body={
                    "size": top_k,
                    "min_score": min_score,
                    "query": script_query,
                    "_source": ["original_id", "job_title"]
                }
            )
            
            # Get original documents
            original_docs = []
            for hit in vector_results['hits']['hits']:
                original_id = hit['_source']['original_id']
                try:
                    original_doc = self.es.get(
                        index=self.original_index, 
                        id=original_id
                    )
                    doc_with_score = {
                        "original_doc": original_doc['_source'],
                        "score": hit['_score']
                    }
                    original_docs.append(doc_with_score)
                except Exception as e:
                    logger.warning(f"Could not fetch original document {original_id}: {str(e)}")
                    continue
            
            search_time = time.time() - start_time
            logger.info(f"Vector search took {search_time:.3f} seconds, found {len(original_docs)} results")
            
            return original_docs
            
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            raise

    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 5,
        text_boost: float = 0.4,
        vector_boost: float = 0.6,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Combine traditional text search with vector search
        """
        try:
            start_time = time.time()
            
            # Traditional text search
            text_query = {
                "multi_match": {
                    "query": query_text,
                    "fields": ["job_title", "synonyms"],
                    "fuzziness": "AUTO"
                }
            }
            
            text_results = self.es.search(
                index=self.original_index,
                body={
                    "size": top_k,
                    "query": text_query
                }
            )
            
            # Vector search
            vector_results = self.vector_search(query_text, top_k)
            
            # Combine and deduplicate results
            combined_results = {}
            
            # Add text search results
            for hit in text_results['hits']['hits']:
                doc_id = hit['_id']
                combined_results[doc_id] = {
                    "doc": hit['_source'],
                    "text_score": hit['_score'],
                    "vector_score": 0
                }
            
            # Add vector search results
            for result in vector_results:
                doc_id = result['original_doc'].get('_id')
                if doc_id in combined_results:
                    combined_results[doc_id]['vector_score'] = result['score']
                else:
                    combined_results[doc_id] = {
                        "doc": result['original_doc'],
                        "text_score": 0,
                        "vector_score": result['score']
                    }
            
            # Calculate final scores
            for doc_id, result in combined_results.items():
                result['final_score'] = (
                    text_boost * result['text_score'] + 
                    vector_boost * result['vector_score']
                )
            
            # Filter by minimum score
            filtered_results = {
                doc_id: result 
                for doc_id, result in combined_results.items() 
                if result['final_score'] >= min_score
            }
            
            # Sort by final score and return top_k
            sorted_results = sorted(
                filtered_results.values(),
                key=lambda x: x['final_score'],
                reverse=True
            )[:top_k]
            
            search_time = time.time() - start_time
            logger.info(f"Hybrid search took {search_time:.3f} seconds, found {len(sorted_results)} results")
            
            return sorted_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {str(e)}")
            raise

    def filtered_vector_search(
        self,
        query_text: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        search_type: str = "combined",
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Vector search with additional filters
        """
        try:
            # Encode query
            query_vector = self.encode_text(query_text)
            vector_field = f"{search_type}_vector"
            
            # Build filter query
            filter_conditions = []
            for field, value in filters.items():
                if isinstance(value, list):
                    filter_conditions.append({"terms": {field: value}})
                else:
                    filter_conditions.append({"term": {field: value}})
            
            # Prepare script query with filters
            search_query = {
                "size": top_k,
                "min_score": min_score,
                "query": {
                    "script_score": {
                        "query": {
                            "bool": {
                                "must": filter_conditions
                            }
                        },
                        "script": {
                            "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                            "params": {"query_vector": query_vector.tolist()}
                        }
                    }
                }
            }
            
            # Execute search
            results = self.es.search(
                index=self.vector_index,
                body=search_query
            )
            
            return results['hits']['hits']
            
        except Exception as e:
            logger.error(f"Error in filtered vector search: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        # Initialize search system
        search = JobTitleSearch()
        
        # Example searches
        test_queries = [
            "ceo",
        ]
        
        for query in test_queries:
            print(f"\nSearching for: {query}")
            
            # Vector search
            print("\nVector Search Results:")
            vector_results = search.vector_search(query, top_k=3)
            for result in vector_results:
                print(f"Score: {result['score']:.3f} - Title: {result['original_doc']['job_title']}")
            
            # Hybrid search
            print("\nHybrid Search Results:")
            hybrid_results = search.hybrid_search(query, top_k=3)
            for result in hybrid_results:
                print(f"Final Score: {result['final_score']:.3f}")
                print(f"Title: {result['doc']['job_title']}")
                print(f"Text Score: {result['text_score']:.3f}")
                print(f"Vector Score: {result['vector_score']:.3f}")
                
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
