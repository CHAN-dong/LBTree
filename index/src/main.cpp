#include <iostream>
#include <chrono>
#include "VocabIO.h"
#include "WorkloadIO.h"
#include "BitmapIndexBaseline.h"
#include "LBTree.h"

int main() {

  // paths
  const std::string vocab_path = "../tree_out/vocab.json";

  // const std::string dataset_path = "/root/dong_11.02/dataset/dblp/dblp_title.csv";
  // const std::string workload_path = "/root/dong_11.02/dataset/dblp/dblp_10venues/workloads/test_all.csv";
  // const std::string py_tree_path = "../tree_out/dblp_top10/";
  // const std::string py_breaked_tree_path = "../tree_out_break/dblp_top10/";

  // const std::string dataset_path = "/root/dong_11.02/dataset/dblp/dblp_title.csv";
  // const std::string workload_path = "/root/dong_11.02/dataset/dblp/dblp_top3/workloads/test_all.csv";
  // const std::string py_tree_path = "../tree_out/dblp_top3/";
  // const std::string py_breaked_tree_path = "../tree_out_break/dblp_top3/";

  // const std::string dataset_path = "/root/dong_11.02/dataset/msmarco/msmarco.csv";
  // const std::string workload_path = "/root/dong_11.02/dataset/msmarco/workload.csv";
  // const std::string py_tree_path = "../tree_out/msmarco/";
  // const std::string py_breaked_tree_path = "../tree_out_break/msmarco/";

  // const std::string dataset_path = "/root/dong_11.02/dataset/ir_datasets/ir_datasets.csv";
  // const std::string workload_path = "/root/dong_11.02/dataset/ir_datasets/workload.csv";
  // const std::string py_tree_path = "../tree_out/ir_datasets/";
  // const std::string py_breaked_tree_path = "../tree_out_break/ir_datasets/";

  const std::string dataset_path = "/root/dong_11.02/dataset/synthetic/synthetic_dataset.csv";
  const std::string workload_path = "/root/dong_11.02/dataset/synthetic/synthetic_query.csv";
  const std::string py_tree_path = "../tree_out/synthetic/";
  const std::string py_breaked_tree_path = "../tree_out_break/synthetic/";
  
  // 1.1) load vocab.json: token -> id
  auto vocab = load_vocab_json(vocab_path);
  std::cout << "Loaded vocab size: " << vocab.size() << "\n";

  // 1.2) load workload.csv: tokens -> ids
  QueryStat ioStat;
  auto queries = load_workload_tokens_to_ids(
      workload_path,
      vocab,
      /*drop_unknown_tokens=*/true,
      /*drop_empty_queries=*/true,
      &ioStat
  );
  std::cout << "Loaded queries: " << queries.size() << "\n";
  std::cout << "Unknown tokens skipped: " << ioStat.unknown_tokens << "\n";
  std::cout << std::endl;

  // Test 1) baseline 
  // std::cout << "Base line ----" << std::endl;
  // BitmapIndexBaseline baseline;
  // auto t0 = std::chrono::steady_clock::now();
  // baseline.buildFromDatasetCSV(dataset_path, vocab, /*drop_unknown_tokens=*/true);
  // auto t1= std::chrono::steady_clock::now();
  // auto construct_time = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  // uint64_t baseline_bytes = baseline.estimateIndexSizeBytes();
  // std::cout << "baseline construct time:" << construct_time / 1e9 << "s" << "\n";
  // std::cout << "Baseline built: #obj=" << baseline.numObjects()
  //           << " #kw_indexed=" << baseline.numIndexedKeywords() << "\n";
  // auto baseline_stat = baseline.benchmark(queries, /*warmup_rounds=*/2);
  // std::cout << "Baseline dataset size MB    = " << (double)baseline_bytes / (1024.0 * 1024.0) << "\n";
  // // std::cout << "Total time(ns): " << baseline_stat.total_ns << "\n";
  // std::cout << "Avg time(ns): " << baseline_stat.avg_ns << "\n";
  // // std::cout << "Total results returned: " << baseline_stat.total_results << "\n";
  // std::cout << std::endl;

  // Test 2) binary tree 
  LBTree binary_tree;
  binary_tree.buildFromPythonTree(py_tree_path);
  // binary_tree.printSummary();
  auto binary_tree_stat = benchmark_workload(binary_tree, queries, /*warmup=*/10);
  auto s_binary = binary_tree.estimateIndexSizeBreakdown();
  std::cout << "Leaf MB        = " << (double)s_binary.leafBytes() / 1024.0 / 1024.0 << "\n";
  std::cout << "Internal MB    = " << (double)s_binary.internalBytes() / 1024.0 / 1024.0 << "\n";
  std::cout << "Total MB       = " << (double)s_binary.totalBytes() / 1024.0 / 1024.0 << "\n";
  // std::cout << "Binary_tree Total time(ns): " << binary_tree_stat.total_ns << "\n";
  std::cout << "Binary_tree Avg time(ns): " << binary_tree_stat.avg_ns << "\n";
  std::cout << "Binary_tree Internal Node Avg time(ns): " << binary_tree_stat.avg_inter_ns << "\n";
  std::cout << "Binary_tree Leaf Node Avg time(ns): " << binary_tree_stat.avg_leaf_ns << "\n";
  // std::cout << "Binary_tree Total results returned: " << binary_tree_stat.total_results << "\n";
  std::cout << std::endl;

  // Test 3) node be breaked tree 
  LBTree fanout_tree;
  fanout_tree.buildFromPythonTree(py_breaked_tree_path);
  // fanout_tree.printSummary();
  auto fanout_tree_stat = benchmark_workload(fanout_tree, queries, /*warmup=*/10);
  auto s_fanout = fanout_tree.estimateIndexSizeBreakdown();
  std::cout << "Leaf MB        = " << (double)s_fanout.leafBytes() / 1024.0 / 1024.0 << "\n";
  std::cout << "Internal MB    = " << (double)s_fanout.internalBytes() / 1024.0 / 1024.0 << "\n";
  std::cout << "Total MB       = " << (double)s_fanout.totalBytes() / 1024.0 / 1024.0 << "\n";
  // std::cout << "Fanout_tree Total time(ns): " << fanout_tree_stat.total_ns << "\n";
  std::cout << "Fanout_tree Avg time(ns): " << fanout_tree_stat.avg_ns << "\n";
  std::cout << "Binary_tree Internal Node Avg time(ns): " << fanout_tree_stat.avg_inter_ns << "\n";
  std::cout << "Binary_tree Leaf Node Avg time(ns): " << fanout_tree_stat.avg_leaf_ns << "\n";
  // std::cout << "Fanout_tree Total results returned: " << fanout_tree_stat.total_results << "\n";

  return 0;
}
