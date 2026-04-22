import itertools
from typing import Dict, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection


class OpenSearchDataIngestor:

    def __init__(self, endpoint: str, port: str, http_auth: Tuple[str, str] = None, 
                 timeout: int = 300, language: str = "english"):
        """
        Initialize OpenSearch data ingestor.
        
        Parameters
        ----------
        endpoint: str
            OpenSearch host endpoint
        port: str
            OpenSearch port
        http_auth: Tuple[str, str], optional
            Tuple of (username, password) for HTTP basic authentication
        timeout: int
            Connection timeout in seconds
        language: str
            Language for text processing
        """
        client_config = {
            'hosts': [{
                'host': endpoint,
                'port': port
            }],
            'use_ssl': True,
            'verify_certs': True,
            'connection_class': RequestsHttpConnection,
            'timeout': timeout,
            'max_retries': 3,
            'retry_on_timeout': True,
        }
        
        if http_auth is not None:
            client_config['http_auth'] = http_auth
        
        self.opensearch = OpenSearch(**client_config)
        self.bulk_size = 50
        self.language = language

    def ingest(self, corpus: Dict[str, Dict[str, str]], index: str):
        total = len(corpus)
        corpus_keys = list(corpus.keys())

        for i in range(0, total, self.bulk_size):
            batch_keys = corpus_keys[i:i + self.bulk_size]

            actions = []
            for key_id in batch_keys:
                doc = corpus[key_id]
                title = doc.get("title", "")
                text = doc.get("text", "")

                # Combine title + text into single passage, no truncation
                passage = (title + " " + text).strip() if title else text.strip()

                actions.append({"index": {"_index": index, "_id": key_id}})
                actions.append({"passage_text": passage})

            try:
                response = self.opensearch.bulk(
                    index=index,
                    body=actions
                )
                if response.get("errors", False):
                    error_count = sum(
                        1 for item in response["items"]
                        if "error" in item.get("index", {})
                    )
                    print(f"Ingested {min(i + self.bulk_size, total)}/{total} ({error_count} errors)")
                    # Log first error for debugging
                    for item in response["items"]:
                        err = item.get("index", {}).get("error")
                        if err:
                            print(f"  Error sample: {err}")
                            break
                else:
                    print(f"Ingested {min(i + self.bulk_size, total)}/{total}")

            except Exception as e:
                print(f"ERROR at batch {i}-{i + self.bulk_size}: {e}")
                raise