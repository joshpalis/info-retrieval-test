import itertools
import textwrap
from typing import Type, List, Dict, Union, Tuple
from opensearchpy import OpenSearch, RequestsHttpConnection


class OpenSearchDataIngestor:

    def __init__(self, endpoint: str, port: str, http_auth: Tuple[str, str] = None, timeout: int = 30, language: str = "english"):
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
            'timeout': timeout
        }
        
        # Add authentication if provided
        if http_auth is not None:
            client_config['http_auth'] = http_auth
        
        self.opensearch = OpenSearch(**client_config)
        self.bulk_size = 200
        self.max_tokens = 512
        self.language = language

    def ingest(self, corpus: Dict[str, Dict[str, str]], index: str):
        '''for i in range(0, 200, self.bulk_size):'''
        for i in range(0, len(corpus), self.bulk_size):
            key_list = itertools.islice(corpus.keys(), i, i + self.bulk_size)

            def get_doc_text(full_string: str):
                str_as_list = textwrap.wrap(full_string, self.max_tokens, break_long_words=False,
                                            break_on_hyphens=False)
                return full_string if len(str_as_list) == 0 else str_as_list[0]
                # return ' '.join(full_string.split()[:500])

            def cleanup(s):
                '''cleaned = s.replace('"', '')
                cleaned = cleaned.replace("\'", "")
                return cleaned
                '''
                return s

            def get_content(corpus_doc):
                if 'title' in corpus_doc.keys():
                    return {
                        'passage_text': cleanup(get_doc_text((corpus_doc["title"] + ' ' + corpus_doc["text"]).strip())), 'text_key': cleanup(corpus_doc['text']), 'title_key': cleanup(corpus_doc['title'])}
                else:
                    return {'passage_text': cleanup(get_doc_text((corpus_doc["text"]).strip())), 'text_key': cleanup(corpus_doc['text'])}

            actions = []
            _ = [
                actions.extend(
                    [{'index': {'_index': index, '_id': key_id}},
                     get_content(corpus[key_id])])
                for key_id in key_list
            ]
            # actions[1::2] = [{'passage_text': corpus[key_id]['text']} for key_id in key_list]
            self.opensearch.bulk(
                index=index,
                body=actions)

            if i % 1000 == 0:
                print("Ingested " + str(i) + " documents")