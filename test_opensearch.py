from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.hybrid.evaluation import EvaluateRetrieval
from beir.hybrid.search import RetrievalOpenSearch
from beir.agentic.search import RetrievalOpenSearchAgentic
from beir.hybrid.data_ingestor import OpenSearchDataIngestor
from bright_converter import convert_bright_task, BRIGHTConverter

import gc
import json
import math
import logging
import pathlib, os, getopt, sys
from datetime import datetime

# SigV4 imports
import boto3
from requests_aws4auth import AWS4Auth


def get_sigv4_auth(region, service='es'):
    """
    Create AWS SigV4 authentication object using boto3 credentials.
    """
    credentials = boto3.Session().get_credentials()
    aws_auth = AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )
    return aws_auth


def load_replacement_queries(file_path):
    """
    Load replacement queries from a JSONL file.
    Each line should be a JSON object with at least '_id' and 'text' fields.
    """
    replacement_queries = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                query_id = obj['_id']
                query_text = obj['text']
                replacement_queries[query_id] = query_text
            except (json.JSONDecodeError, KeyError) as e:
                logging.warning(f"Skipping line {line_num} in {file_path}: {e}")
    logging.info(f"Loaded {len(replacement_queries)} replacement queries from {file_path}")
    return replacement_queries


def compute_per_query_ndcg(qrels, results, k_values):
    """
    Compute NDCG@k for each individual query.

    Returns:
        dict: {query_id: {"NDCG@k1": score, "NDCG@k2": score, ...}}
    """
    per_query_scores = {}

    for qid in results:
        if qid not in qrels:
            continue

        per_query_scores[qid] = {}

        for k in k_values:
            sorted_scores = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:k]

            dcg = 0.0
            for i, (doc_id, _) in enumerate(sorted_scores):
                rel = qrels[qid].get(doc_id, 0)
                dcg += (2 ** rel - 1) / math.log2(i + 2)

            ideal_rels = sorted(qrels[qid].values(), reverse=True)[:k]
            idcg = 0.0
            for i, rel in enumerate(ideal_rels):
                idcg += (2 ** rel - 1) / math.log2(i + 2)

            ndcg = dcg / idcg if idcg > 0 else 0.0
            per_query_scores[qid][f'NDCG@{k}'] = round(ndcg, 5)

    return per_query_scores


def write_jsonl(file_path, records):
    """Write a list of dicts to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    print(f"Wrote {len(records)} records to {file_path}")


def write_results_file(index_name, evaluation_results):
    """
    Write all evaluation results to a JSON file in the results directory.
    File is named: {index_name}_{timestamp}.json
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "results")
    os.makedirs(results_dir, exist_ok=True)

    filename = f'{index_name}_{timestamp}.json'
    filepath = os.path.join(results_dir, filename)

    output = {
        'index': index_name,
        'timestamp': timestamp,
        'evaluations': evaluation_results
    }

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f'\nResults file written: {filepath}')
    return filepath


def main(argv):
    opts, args = getopt.getopt(argv, "d:u:h:p:i:m:n:o:l:e:a:w:r:s:q:t:",
                               ["dataset=", "dataset_url=", "os_host=", "os_port=", "os_index=", "os_model_id=",
                                "num_of_runs=", "operation=", "pipelines=", "method=", "username=", "password=",
                                "region=", "sigv4", "service=", "queries_file=",
                                "bright_task=", "dataset_type=", "all_bright"])
    dataset = ''
    url = ''
    endpoint = ''
    port = ''
    index = ''
    model_id = ''
    num_of_runs = 1
    operation = "both"
    pipelines = ''
    mmethod = 'hybrid'
    username = ''
    password = ''
    region = ''
    use_sigv4 = False
    service = 'es'
    queries_file = ''
    bright_task = ''
    dataset_type = 'beir'
    all_bright = False

    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            dataset = arg
        elif opt in ("-u", "--dataset_url"):
            url = arg
        elif opt in ("-h", "--os_host"):
            endpoint = arg
        elif opt in ("-p", "--os_port"):
            port = arg
        elif opt in ("-i", "--os_index"):
            index = arg
        elif opt in ("-m", "--os_model_id"):
            model_id = arg
        elif opt in ("-n", "--num_of_runs"):
            num_of_runs = int(arg)
        elif opt in ("-o", "--operation"):
            operation = arg
        elif opt in ("-l", "--pipelines"):
            pipelines = arg
        elif opt in ("-e", "--method"):
            mmethod = arg
        elif opt in ("-a", "--username"):
            username = arg
        elif opt in ("-w", "--password"):
            password = arg
        elif opt in ("-r", "--region"):
            region = arg
        elif opt in ("--sigv4",):
            use_sigv4 = True
        elif opt in ("-s", "--service"):
            service = arg
        elif opt in ("-q", "--queries_file"):
            queries_file = arg
        elif opt in ("-t", "--bright_task"):
            bright_task = arg
            dataset_type = 'bright'
        elif opt in ("--dataset_type",):
            dataset_type = arg
        elif opt in ("--all_bright",):
            all_bright = True
            dataset_type = 'bright'

    # Auth setup
    if use_sigv4:
        if not region:
            raise ValueError("Region is required for SigV4 authentication. Use --region flag.")
        auth = get_sigv4_auth(region, service)
        logging.info(f"Using SigV4 authentication for region: {region}, service: {service}")
    elif username and password:
        auth = (username, password)
        logging.info("Using basic authentication")
    else:
        auth = None
        logging.info("No authentication configured")

    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])

    # ---- Determine which tasks to run ----
    if all_bright:
        tasks = BRIGHTConverter.AVAILABLE_TASKS
        logging.info(f"Running all {len(tasks)} BRIGHT tasks: {tasks}")
    elif dataset_type == 'bright':
        if not bright_task:
            raise ValueError(
                f"BRIGHT task is required. Use --bright_task=<task_name> or --all_bright. "
                f"Available tasks: {BRIGHTConverter.AVAILABLE_TASKS}"
            )
        tasks = [bright_task]
    else:
        tasks = []

    # ---- BRIGHT path: loop over tasks ----
    if tasks:
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        all_results = {}

        for task in tasks:
            task_index = index if index else f"bright-{task}"
            logging.info("=" * 60)
            logging.info(f"BRIGHT TASK: {task} | Index: {task_index}")
            logging.info("=" * 60)

            try:
                # Convert BRIGHT HF data to BEIR format on disk
                data_path = convert_bright_task(task, out_dir)
                logging.info(f"BRIGHT data converted to BEIR format at: {data_path}")

                # Load using existing GenericDataLoader
                corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

                logging.info(f"Corpus size: {len(corpus)}")
                logging.info(f"Number of queries: {len(queries)}")
                logging.info(f"Qrel judgments: {sum(len(v) for v in qrels.values())} across {len(qrels)} queries")

                if operation == 'ingest' or operation == 'both':
                    ingest_data(corpus, endpoint, task_index, port, auth)
                    logging.info(f"Ingestion complete for task: {task}")

                if operation == 'evaluate' or operation == 'both':
                    evaluate(corpus, endpoint, task_index, model_id, port, qrels, queries,
                             num_of_runs, pipelines, mmethod, auth)

                all_results[task] = {"status": "success", "corpus_size": len(corpus), "queries": len(queries)}

            except Exception as e:
                logging.error(f"Failed on task {task}: {e}")
                all_results[task] = {"status": "failed", "error": str(e)}

            finally:
                # Free memory between tasks
                gc.collect()

        # Print summary
        logging.info("=" * 60)
        logging.info("ALL BRIGHT TASKS SUMMARY")
        logging.info("=" * 60)
        for task, result in all_results.items():
            status = result["status"]
            if status == "success":
                logging.info(f"  ✓ {task:<25} corpus={result['corpus_size']:>8}  queries={result['queries']:>5}")
            else:
                logging.info(f"  ✗ {task:<25} ERROR: {result['error']}")

    # ---- Original BEIR path ----
    else:
        url = url.format(dataset)
        out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "datasets")
        data_path = util.download_and_unzip(url, out_dir)
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        logging.info(f"Corpus size: {len(corpus)}")
        logging.info(f"Number of queries: {len(queries)}")
        logging.info(f"Qrel judgments: {sum(len(v) for v in qrels.values())} across {len(qrels)} queries")

        if operation == 'ingest' or operation == 'both':
            ingest_data(corpus, endpoint, index, port, auth)

        if operation == 'evaluate' or operation == 'both':
            evaluate(corpus, endpoint, index, model_id, port, qrels, queries, num_of_runs, pipelines, mmethod,
                     auth)


def ingest_data(corpus, endpoint, index, port, auth=None):
    OpenSearchDataIngestor(endpoint, port, http_auth=auth).ingest(corpus, index=index)


def evaluate(corpus, endpoint, index, model_id, port, qrels, queries, num_of_runs, pipelines, mmethod, auth=None):
    # This k values are being used for BM25 search
    bm25_k_values = [1, 3, 5, 10, 100]
    # This K values are being used for dense model search
    model_k_values = [1, 3, 5, 10, 100]
    # this k values are being used for scoring
    k_values = [5, 10, 100]

    mm = mmethod.split(',')

    # Accumulate all evaluation results for the results file
    all_evaluation_results = []

    # ──────────────────────────────────────────────
    # evaluate_both – run agentic & bm25, compare per-query NDCG
    # ──────────────────────────────────────────────
    if 'evaluate_both' in mm:
        print('=' * 80)
        print('Starting evaluate_both: agentic + BM25 comparison')
        print('=' * 80)

        result_size = max(bm25_k_values)

        # ---------- Run Agentic search ----------
        print('\n--- Running agentic search ---')
        os_agentic = RetrievalOpenSearchAgentic(
            endpoint,
            port,
            index_name=index,
            search_pipeline=pipelines.split(',')[0],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever_agentic = EvaluateRetrieval(os_agentic, bm25_k_values)
        agentic_results = os_agentic.search_agentic(corpus, queries, top_k=result_size)
        agentic_ndcg, agentic_map, agentic_recall, agentic_precision = retriever_agentic.evaluate(
            qrels, agentic_results, k_values
        )

        # ---------- Run BM25 search ----------
        print('\n--- Running BM25 search ---')
        os_bm25 = RetrievalOpenSearch(
            endpoint,
            port,
            index_name=index,
            model_id=model_id,
            search_method='bm25',
            pipeline_name=pipelines.split(',')[0],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever_bm25 = EvaluateRetrieval(os_bm25, bm25_k_values)
        bm25_results = os_bm25.search_bm25(corpus, queries, top_k=result_size)
        bm25_ndcg, bm25_map, bm25_recall, bm25_precision = retriever_bm25.evaluate(
            qrels, bm25_results, k_values
        )

        # ---------- Compute per-query NDCG then FREE result dicts ----------
        agentic_per_query = compute_per_query_ndcg(qrels, agentic_results, k_values)
        del agentic_results
        gc.collect()
        print('[memory] freed agentic_results')

        bm25_per_query = compute_per_query_ndcg(qrels, bm25_results, k_values)
        del bm25_results
        gc.collect()
        print('[memory] freed bm25_results')

        primary_k = 10 if 10 in k_values else k_values[0]
        primary_key = f'NDCG@{primary_k}'

        # ---------- Build per-query comparison records ----------
        comparison_records = []
        agentic_wins = 0
        bm25_wins = 0
        ties = 0

        for qid in queries:
            agentic_scores = agentic_per_query.get(qid, {f'NDCG@{k}': 0.0 for k in k_values})
            bm25_scores = bm25_per_query.get(qid, {f'NDCG@{k}': 0.0 for k in k_values})

            agentic_primary = agentic_scores.get(primary_key, 0.0)
            bm25_primary = bm25_scores.get(primary_key, 0.0)

            if agentic_primary > bm25_primary:
                winner = 'agentic'
                agentic_wins += 1
            elif bm25_primary > agentic_primary:
                winner = 'bm25'
                bm25_wins += 1
            else:
                winner = 'tie'
                ties += 1

            comparison_records.append({
                '_id': qid,
                'text': queries[qid],
                'agentic_scores': agentic_scores,
                'bm25_scores': bm25_scores,
                'winner': winner,
                'score_diff': round(agentic_primary - bm25_primary, 5)
            })

        # ---------- Print aggregate summary ----------
        print('\n' + '=' * 80)
        print('EVALUATE_BOTH AGGREGATE RESULTS')
        print('=' * 80)

        print('\nAgentic aggregate scores:')
        for k_label, score in agentic_ndcg.items():
            print(f'  {k_label}: {score:.5f}')

        print('\nBM25 aggregate scores:')
        for k_label, score in bm25_ndcg.items():
            print(f'  {k_label}: {score:.5f}')

        print(f'\nPer-query winner summary (based on {primary_key}):')
        print(f'  Agentic wins : {agentic_wins}/{len(comparison_records)}')
        print(f'  BM25 wins    : {bm25_wins}/{len(comparison_records)}')
        print(f'  Ties         : {ties}/{len(comparison_records)}')

        # ---------- Accumulate results ----------
        all_evaluation_results.append({
            'method': 'evaluate_both',
            'agentic_aggregate': {
                'ndcg': agentic_ndcg,
                'map': agentic_map,
                'recall': agentic_recall,
                'precision': agentic_precision
            },
            'bm25_aggregate': {
                'ndcg': bm25_ndcg,
                'map': bm25_map,
                'recall': bm25_recall,
                'precision': bm25_precision
            }
        })

        # Free per-query dicts
        del agentic_per_query, bm25_per_query, comparison_records
        gc.collect()

        print('--- end of results for evaluate_both ---')

    # ──────────────────────────────────────────────
    # evaluate_both_neural – run agentic & neural, compare per-query NDCG
    # ──────────────────────────────────────────────
    if 'evaluate_both_neural' in mm:
        print('=' * 80)
        print('Starting evaluate_both_neural: agentic + neural comparison')
        print('=' * 80)

        top_k = max(model_k_values)
        result_size = max(bm25_k_values)

        # ---------- Run Agentic search ----------
        print('\n--- Running agentic search ---')
        os_agentic = RetrievalOpenSearchAgentic(
            endpoint,
            port,
            index_name=index,
            search_pipeline=pipelines.split(',')[0],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever_agentic = EvaluateRetrieval(os_agentic, model_k_values)
        agentic_results = os_agentic.search_agentic(corpus, queries, top_k=result_size)
        agentic_ndcg, agentic_map, agentic_recall, agentic_precision = retriever_agentic.evaluate(
            qrels, agentic_results, k_values
        )

        # ---------- Run Neural search ----------
        print('\n--- Running neural search ---')
        pipeline_name = pipelines.split(',')[0]
        os_neural = RetrievalOpenSearch(
            endpoint,
            port,
            index_name=index,
            model_id=model_id,
            search_method='neural',
            pipeline_name=pipeline_name,
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever_neural = EvaluateRetrieval(os_neural, model_k_values)
        neural_results = os_neural.search_vector(corpus, queries, top_k=top_k, result_size=result_size)
        neural_ndcg, neural_map, neural_recall, neural_precision = retriever_neural.evaluate(
            qrels, neural_results, k_values
        )

        # ---------- Compute per-query NDCG then FREE result dicts ----------
        agentic_per_query = compute_per_query_ndcg(qrels, agentic_results, k_values)
        del agentic_results
        gc.collect()
        print('[memory] freed agentic_results')

        neural_per_query = compute_per_query_ndcg(qrels, neural_results, k_values)
        del neural_results
        gc.collect()
        print('[memory] freed neural_results')

        primary_k = 10 if 10 in k_values else k_values[0]
        primary_key = f'NDCG@{primary_k}'

        # ---------- Build per-query comparison records ----------
        comparison_records = []
        agentic_wins = 0
        neural_wins = 0
        ties = 0

        for qid in queries:
            agentic_scores = agentic_per_query.get(qid, {f'NDCG@{k}': 0.0 for k in k_values})
            neural_scores = neural_per_query.get(qid, {f'NDCG@{k}': 0.0 for k in k_values})

            agentic_primary = agentic_scores.get(primary_key, 0.0)
            neural_primary = neural_scores.get(primary_key, 0.0)

            if agentic_primary > neural_primary:
                winner = 'agentic'
                agentic_wins += 1
            elif neural_primary > agentic_primary:
                winner = 'neural'
                neural_wins += 1
            else:
                winner = 'tie'
                ties += 1

            comparison_records.append({
                '_id': qid,
                'text': queries[qid],
                'agentic_scores': agentic_scores,
                'neural_scores': neural_scores,
                'winner': winner,
                'score_diff': round(agentic_primary - neural_primary, 5)
            })

        # ---------- Print aggregate summary ----------
        print('\n' + '=' * 80)
        print('EVALUATE_BOTH_NEURAL AGGREGATE RESULTS')
        print('=' * 80)

        print('\nAgentic aggregate scores:')
        for k_label, score in agentic_ndcg.items():
            print(f'  {k_label}: {score:.5f}')

        print('\nNeural aggregate scores:')
        for k_label, score in neural_ndcg.items():
            print(f'  {k_label}: {score:.5f}')

        print(f'\nPer-query winner summary (based on {primary_key}):')
        print(f'  Agentic wins : {agentic_wins}/{len(comparison_records)}')
        print(f'  Neural wins  : {neural_wins}/{len(comparison_records)}')
        print(f'  Ties         : {ties}/{len(comparison_records)}')

        # ---------- Accumulate results ----------
        all_evaluation_results.append({
            'method': 'evaluate_both_neural',
            'agentic_aggregate': {
                'ndcg': agentic_ndcg,
                'map': agentic_map,
                'recall': agentic_recall,
                'precision': agentic_precision
            },
            'neural_aggregate': {
                'ndcg': neural_ndcg,
                'map': neural_map,
                'recall': neural_recall,
                'precision': neural_precision
            }
        })

        # Free per-query dicts
        del agentic_per_query, neural_per_query, comparison_records
        gc.collect()

        print('--- end of results for evaluate_both_neural ---')

    # ──────────────────────────────────────────────
    # evaluate_both_hybrid – run agentic & hybrid, compare per-query NDCG
    # ──────────────────────────────────────────────
    if 'evaluate_both_hybrid' in mm:
        print('=' * 80)
        print('Starting evaluate_both_hybrid: agentic + hybrid comparison')
        print('=' * 80)

        top_k = max(model_k_values)
        result_size = max(bm25_k_values)

        # ---------- Run Agentic search ----------
        print('\n--- Running agentic search ---')
        os_agentic = RetrievalOpenSearchAgentic(
            endpoint,
            port,
            index_name=index,
            search_pipeline=pipelines.split(',')[0],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever_agentic = EvaluateRetrieval(os_agentic, model_k_values)
        agentic_results = os_agentic.search_agentic(corpus, queries, top_k=result_size)
        agentic_ndcg, agentic_map, agentic_recall, agentic_precision = retriever_agentic.evaluate(
            qrels, agentic_results, k_values
        )

        # ---------- Run Hybrid search ----------
        print('\n--- Running hybrid search ---')
        pipeline_name = pipelines.split(',')[0]
        os_hybrid = RetrievalOpenSearch(
            endpoint,
            port,
            index_name=index,
            model_id=model_id,
            search_method='hybrid',
            pipeline_name=pipeline_name,
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever_hybrid = EvaluateRetrieval(os_hybrid, model_k_values)
        hybrid_results = os_hybrid.search_vector(corpus, queries, top_k=top_k, result_size=result_size)
        hybrid_ndcg, hybrid_map, hybrid_recall, hybrid_precision = retriever_hybrid.evaluate(
            qrels, hybrid_results, k_values
        )

        # ---------- Compute per-query NDCG then FREE result dicts ----------
        agentic_per_query = compute_per_query_ndcg(qrels, agentic_results, k_values)
        del agentic_results
        gc.collect()
        print('[memory] freed agentic_results')

        hybrid_per_query = compute_per_query_ndcg(qrels, hybrid_results, k_values)
        del hybrid_results
        gc.collect()
        print('[memory] freed hybrid_results')

        primary_k = 10 if 10 in k_values else k_values[0]
        primary_key = f'NDCG@{primary_k}'

        # ---------- Build per-query comparison records ----------
        comparison_records = []
        agentic_wins = 0
        hybrid_wins = 0
        ties = 0

        for qid in queries:
            agentic_scores = agentic_per_query.get(qid, {f'NDCG@{k}': 0.0 for k in k_values})
            hybrid_scores = hybrid_per_query.get(qid, {f'NDCG@{k}': 0.0 for k in k_values})

            agentic_primary = agentic_scores.get(primary_key, 0.0)
            hybrid_primary = hybrid_scores.get(primary_key, 0.0)

            if agentic_primary > hybrid_primary:
                winner = 'agentic'
                agentic_wins += 1
            elif hybrid_primary > agentic_primary:
                winner = 'hybrid'
                hybrid_wins += 1
            else:
                winner = 'tie'
                ties += 1

            comparison_records.append({
                '_id': qid,
                'text': queries[qid],
                'agentic_scores': agentic_scores,
                'hybrid_scores': hybrid_scores,
                'winner': winner,
                'score_diff': round(agentic_primary - hybrid_primary, 5)
            })

        # ---------- Print aggregate summary ----------
        print('\n' + '=' * 80)
        print('EVALUATE_BOTH_HYBRID AGGREGATE RESULTS')
        print('=' * 80)

        print('\nAgentic aggregate scores:')
        for k_label, score in agentic_ndcg.items():
            print(f'  {k_label}: {score:.5f}')

        print('\nHybrid aggregate scores:')
        for k_label, score in hybrid_ndcg.items():
            print(f'  {k_label}: {score:.5f}')

        print(f'\nPer-query winner summary (based on {primary_key}):')
        print(f'  Agentic wins : {agentic_wins}/{len(comparison_records)}')
        print(f'  Hybrid wins  : {hybrid_wins}/{len(comparison_records)}')
        print(f'  Ties         : {ties}/{len(comparison_records)}')

        # ---------- Accumulate results ----------
        all_evaluation_results.append({
            'method': 'evaluate_both_hybrid',
            'agentic_aggregate': {
                'ndcg': agentic_ndcg,
                'map': agentic_map,
                'recall': agentic_recall,
                'precision': agentic_precision
            },
            'hybrid_aggregate': {
                'ndcg': hybrid_ndcg,
                'map': hybrid_map,
                'recall': hybrid_recall,
                'precision': hybrid_precision
            }
        })

        # Free per-query dicts
        del agentic_per_query, hybrid_per_query, comparison_records
        gc.collect()

        print('--- end of results for evaluate_both_hybrid ---')

    # ──────────────────────────────────────────────
    # Existing: standalone agentic
    # ──────────────────────────────────────────────
    if 'agentic' in mm and 'evaluate_both' not in mm and 'evaluate_both_neural' not in mm and 'evaluate_both_hybrid' not in mm:
        method = 'agentic'
        print('starting search method ' + method)

        agentic_queries = queries
        if len(queries) > 1000:
            print(f'Limiting queries from {len(queries)} to 1000 for agentic search')
            agentic_queries = dict(list(queries.items())[:1000])

        os_retrival = RetrievalOpenSearchAgentic(
            endpoint,
            port,
            index_name=index,
            search_pipeline=pipelines.split(',')[0],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever = EvaluateRetrieval(os_retrival, bm25_k_values)
        result_size = max(bm25_k_values)
        results = os_retrival.search_agentic(corpus, agentic_queries, top_k=result_size)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

        # ---------- Accumulate results ----------
        all_evaluation_results.append({
            'method': 'agentic',
            'aggregate': {
                'ndcg': ndcg,
                'map': _map,
                'recall': recall,
                'precision': precision
            }
        })

        # FREE result dict
        del results
        gc.collect()
        print('[memory] freed agentic results')

        print('--- end of results for ' + method)

    # ──────────────────────────────────────────────
    # Existing: standalone bm25
    # ──────────────────────────────────────────────
    if 'bm25' in mm and 'evaluate_both' not in mm:
        method = 'bm25'
        print('starting search method ' + method)

        os_retrival = RetrievalOpenSearch(
            endpoint,
            port,
            index_name=index,
            model_id=model_id,
            search_method=method,
            pipeline_name=pipelines.split(',')[0],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True
        )
        retriever = EvaluateRetrieval(os_retrival, bm25_k_values)
        result_size = max(bm25_k_values)
        results = os_retrival.search_bm25(corpus, queries, top_k=result_size)
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)

        # ---------- Accumulate results ----------
        all_evaluation_results.append({
            'method': 'bm25',
            'aggregate': {
                'ndcg': ndcg,
                'map': _map,
                'recall': recall,
                'precision': precision
            }
        })

        # FREE result dict
        del results
        gc.collect()
        print('[memory] freed bm25 results')

        print('--- end of results for ' + method)

    # ──────────────────────────────────────────────
    # Existing: vector-based methods (neural, hybrid, bool)
    # ──────────────────────────────────────────────
    for method in get_vector_methods(mm):
        for pipeline in pipelines.split(','):
            print('starting search method ' + method + " for pipeline " + pipeline)
            os_retrival = RetrievalOpenSearch(endpoint, port,
                                              index_name=index,
                                              model_id=model_id,
                                              search_method=method,
                                              pipeline_name=pipeline,
                                              http_auth=auth)
            retriever = EvaluateRetrieval(os_retrival, model_k_values)
            top_k = max(model_k_values)
            result_size = max(bm25_k_values)
            all_experiments_took_time = []
            last_ndcg = None
            last_map = None
            last_recall = None
            last_precision = None
            for run in range(0, num_of_runs):
                results = os_retrival.search_vector(corpus, queries, top_k=top_k, result_size=result_size)
                # ── Collect took_time from each run ──────────────
                all_experiments_took_time.append(os_retrival.took_time.copy())
                # ─────────────────────────────────────────────────
                last_ndcg, last_map, last_recall, last_precision = retriever.evaluate(qrels, results, k_values)

                # FREE result dict after each run
                del results
                gc.collect()

            retriever.evaluate_time(all_experiments_took_time)

            # ---------- Accumulate results (from last run) ----------
            all_evaluation_results.append({
                'method': method,
                'pipeline': pipeline,
                'num_runs': num_of_runs,
                'aggregate': {
                    'ndcg': last_ndcg,
                    'map': last_map,
                    'recall': last_recall,
                    'precision': last_precision
                }
            })

            print('[memory] freed vector search results')
            print('--- end of results for ' + method + " and pipeline " + pipeline)

    # ──────────────────────────────────────────────
    # Write all accumulated results to a single file
    # ──────────────────────────────────────────────
    if all_evaluation_results:
        write_results_file(index, all_evaluation_results)


def get_vector_methods(mm):
    vector_methods = []
    if 'neural' in mm:
        vector_methods.append('neural')
    if 'hybrid' in mm:
        vector_methods.append('hybrid')
    if 'bool' in mm:
        vector_methods.append('bool')
    return vector_methods


if __name__ == "__main__":
    main(sys.argv[1:])