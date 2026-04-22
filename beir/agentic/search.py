import logging
import textwrap
import random
import os
import json
from typing import Dict, List, Optional
from opensearchpy import OpenSearch, RequestsHttpConnection
from datetime import datetime

logger = logging.getLogger(__name__)


class RetrievalOpenSearchAgentic:

    def __init__(self, endpoint: str, port: str, index_name: str, 
                 search_pipeline: str = 'agentic_search_pipeline',
                 query_fields: Optional[List[str]] = None,
                 batch_size: int = 128, 
                 corpus_chunk_size: int = 50000, 
                 timeout: int = 520,
                 http_auth=None,
                 use_ssl: bool = True,
                 verify_certs: bool = True,
                 **kwargs):
        self.took_time = {}
        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.show_progress_bar = True
        self.convert_to_tensor = True
        self.results = {}
        self.index_name = index_name
        self.search_pipeline = search_pipeline
        self.query_fields = query_fields if query_fields else ['passage_text']
        self.max_tokens = 512

        # Create results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"[DEBUG] Initializing OpenSearch client...")
        print(f"[DEBUG] Endpoint: {endpoint}:{port}")
        print(f"[DEBUG] Index: {index_name}")
        print(f"[DEBUG] Search Pipeline: {search_pipeline}")
        print(f"[DEBUG] Auth type: {type(http_auth)}")
        print(f"[DEBUG] Results directory: {os.path.abspath(self.results_dir)}")

        self.opensearch = OpenSearch(
            hosts=[{
                'host': endpoint,
                'port': port
            }],
            http_auth=http_auth,
            use_ssl=use_ssl,      
            verify_certs=verify_certs, 
            connection_class=RequestsHttpConnection,
            timeout=timeout
        )

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test the OpenSearch connection"""
        try:
            info = self.opensearch.info()
            print(f"[DEBUG] Connected to OpenSearch version: {info['version']['number']}")
            
            # Check if index exists
            if self.opensearch.indices.exists(index=self.index_name):
                print(f"[DEBUG] Index '{self.index_name}' exists")
                count = self.opensearch.count(index=self.index_name)
                print(f"[DEBUG] Document count: {count['count']}")
            else:
                print(f"[ERROR] Index '{self.index_name}' does NOT exist!")
                
        except Exception as e:
            print(f"[ERROR] Failed to connect to OpenSearch: {e}")
            raise

    def _save_query_and_dsl(self, query_id: str, query_text: str, response: dict):
        """Extract dsl_query from the response and save it alongside the query text."""
        try:
            ext = response.get("ext", {})
            dsl_query_raw = ext.get("dsl_query")
            agent_steps = ext.get("agent_steps_summary")

            # Parse dsl_query string into a dict if it's a JSON string
            if isinstance(dsl_query_raw, str):
                try:
                    dsl_query = json.loads(dsl_query_raw)
                except json.JSONDecodeError:
                    dsl_query = dsl_query_raw
            else:
                dsl_query = dsl_query_raw

            record = {
                "query_id": query_id,
                "query_text": query_text,
                "dsl_query": dsl_query,
                "agent_steps_summary": agent_steps,
                "took_ms": response.get("took"),
                "total_hits": response.get("hits", {}).get("total", {}).get("value")
            }

            # Sanitize query_id for use as a filename
            safe_id = str(query_id).replace("/", "_").replace("\\", "_")
            filepath = os.path.join(self.results_dir, f"query_{safe_id}.json")

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[WARNING] Failed to save query/DSL for query_id={query_id}: {e}")

    def search_agentic(self,
                      corpus: Dict[str, Dict[str, str]],
                      queries: Dict[str, str],
                      top_k: int,
                      return_sorted: bool = False,
                      verbose: bool = False,
                      **kwargs) -> Dict[str, Dict[str, float]]:
        
        def get_doc_text(full_string: str):
            """Truncate text to max tokens"""
            str_as_list = textwrap.wrap(full_string, self.max_tokens, 
                                       break_long_words=False,
                                       break_on_hyphens=False)
            return full_string if len(str_as_list) == 0 else str_as_list[0]

        def get_body_agentic(query_text: str):
            """Build agentic search query body"""
            return {
                'size': top_k,
                "_source": {
                    "excludes": ["passage_embedding"]
                },
                'query': {
                    'agentic': {
                        'query_text': query_text,
                        'query_fields': self.query_fields
                    }
                }
            }

        print(f"[DEBUG] Starting Agentic Search...")
        print(f"[DEBUG] Search Pipeline: {self.search_pipeline}")
        print(f"[DEBUG] Index: {self.index_name}")
        print(f"[DEBUG] Query Fields: {self.query_fields}")
        print(f"[DEBUG] Top K: {top_k}")
        print(f"[DEBUG] Number of queries: {len(queries)}")
        
        query_ids = list(queries.keys())
        self.results = {qid: {} for qid in query_ids}
        self.took_time = {qid: {} for qid in query_ids}
        queries_list = [queries[qid] for qid in queries]

        # Test with a single query first
        print(f"[DEBUG] Testing single query...")
        test_query = queries_list[0]
        test_body = get_body_agentic(get_doc_text(test_query))
        print(f"[DEBUG] Test query text: {test_query[:100]}...")
        print(f"[DEBUG] Test query body: {test_body}")
        
        try:
            test_response = self.opensearch.search(
                index=self.index_name,
                body=test_body,
                params={"search_pipeline": self.search_pipeline}
            )
            print(f"[DEBUG] Test query successful!")
            print(f"[DEBUG] Test response hits: {len(test_response.get('hits', {}).get('hits', []))}")
            print(f"[DEBUG] Test response took: {test_response.get('took', 'N/A')} ms")

        except Exception as e:
            print(f"[ERROR] Test query failed: {e}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            # Try without the search pipeline to see if that's the issue
            print(f"[DEBUG] Trying without search_pipeline...")
            try:
                test_response_no_pipeline = self.opensearch.search(
                    index=self.index_name,
                    body=test_body
                )
                print(f"[DEBUG] Query without pipeline succeeded - pipeline might be the issue")
            except Exception as e2:
                print(f"[ERROR] Query without pipeline also failed: {e2}")
            raise

        print(f"[DEBUG] Sorting Corpus by document length (Longest first)...")
        corpus_ids = sorted(corpus, 
                          key=lambda k: len(corpus[k].get("title", "") + corpus[k].get("text", "")),
                          reverse=True)
        corpus = [corpus[cid] for cid in corpus_ids]

        # Execute actual queries
        query_responses = []
        failed_queries = 0
        
        for i in range(len(query_ids)):
            query_id = query_ids[i]
            q = queries_list[i]
            
            try:
                response = self.opensearch.search(
                    index=self.index_name,
                    body=get_body_agentic(get_doc_text(q)),
                    params={"search_pipeline": self.search_pipeline}
                )
                
                query_responses.append(response)

                # ---- Save query text + generated DSL to results/ ----
                #self._save_query_and_dsl(query_id, q, response)
                
                if i % 10 == 0:
                    print(f"[DEBUG] Executed queries: {i}/{len(query_ids)}")
                    
            except Exception as e:
                print(f"[ERROR] Query {i} (ID: {query_id}) failed: {e}")
                failed_queries += 1
                query_responses.append({})

        print(f"[DEBUG] Completed queries. Failed: {failed_queries}/{len(query_ids)}")

        # ---- Write a combined summary file with ALL queries ----
        # self._save_all_queries_summary(query_ids, queries_list, query_responses)

        # Parse responses and build results
        for i in range(len(query_responses)):
            query_id = query_ids[i]
            response = query_responses[i]
            
            if not response:
                continue
            
            try:
                if 'hits' in response and 'hits' in response['hits']:
                    hits = response['hits']['hits']
                    took = response.get('took', 0)
                    
                    for hit in hits:
                        corp_id = hit.get('_id')
                        score = hit.get('_score', 0.0)
                        
                        if corp_id and corp_id != query_id:
                            self.results[query_id][corp_id] = score
                    
                    self.took_time[query_id] = took
                else:
                    print(f"[WARNING] Unexpected response format for query {query_id}")
                
            except Exception as e:
                print(f"[ERROR] Failed to parse response for query {query_id}: {e}")

        print(f"[DEBUG] Results collected for {len([r for r in self.results.values() if r])} queries")
        return self.results

    def _save_all_queries_summary(self, query_ids, queries_list, query_responses):
        """Write a single summary file containing all query texts and their generated DSL queries."""
        summary = []
        for i, (qid, qtxt, resp) in enumerate(zip(query_ids, queries_list, query_responses)):
            if not resp:
                summary.append({
                    "query_id": qid,
                    "query_text": qtxt,
                    "dsl_query": None,
                    "error": "empty response"
                })
                continue

            ext = resp.get("ext", {})
            dsl_raw = ext.get("dsl_query")
            if isinstance(dsl_raw, str):
                try:
                    dsl_raw = json.loads(dsl_raw)
                except json.JSONDecodeError:
                    pass

            summary.append({
                "query_id": qid,
                "query_text": qtxt,
                "dsl_query": dsl_raw,
                "agent_steps_summary": ext.get("agent_steps_summary"),
                "took_ms": resp.get("took"),
                "total_hits": resp.get("hits", {}).get("total", {}).get("value")
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = os.path.join(self.results_dir, f"{self.index_name}_queries_summary_{timestamp}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Saved combined summary -> {summary_path}")

    def get_timing_stats(self):
        """Get timing statistics for executed queries"""
        if not self.took_time:
            return {}
        
        times = [t for t in self.took_time.values() if isinstance(t, (int, float)) and t > 0]
        if not times:
            return {}
        
        return {
            'mean_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'total_ms': sum(times),
            'count': len(times)
        }

    def print_timing_stats(self):
        """Print formatted timing statistics"""
        stats = self.get_timing_stats()
        if stats:
            print("=== Agentic Search Timing Statistics ===")
            print(f"Total Queries: {stats['count']}")
            print(f"Mean Time: {stats['mean_ms']:.2f} ms")
            print(f"Min Time: {stats['min_ms']} ms")
            print(f"Max Time: {stats['max_ms']} ms")
            print(f"Total Time: {stats['total_ms']} ms")
        else:
            print("No timing statistics available")