#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class LBTree;

struct QueryStat {
    int64_t num_queries = 0;
    int64_t total_ns = 0;
    int64_t internal_ns = 0;
    int64_t leaf_ns = 0;
    double avg_ns = 0.0;
    double avg_inter_ns = 0.0;
    double avg_leaf_ns = 0.0;
    int64_t total_results = 0;
    int64_t skipped_queries = 0;
    int64_t unknown_tokens = 0;
};

std::vector<std::vector<int>> load_workload_tokens_to_ids(
    const std::string& csv_path,
    const std::unordered_map<std::string, int>& vocab,
    bool drop_unknown_tokens = true,
    bool drop_empty_queries = true,
    QueryStat* optional_stat = nullptr
);

QueryStat benchmark_workload(
    const LBTree& tree,
    const std::vector<std::vector<int>>& queries,
    int warmup = 1
);
