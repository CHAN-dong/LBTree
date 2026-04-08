#include "WorkloadIO.h"
#include "LBTree.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <cctype>

static std::string trim(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace((unsigned char)s[a])) ++a;
  size_t b = s.size();
  while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
  return s.substr(a, b - a);
}

static void normalize_separators(std::string& s) {
  for (char& c : s) {
    if (c == ',' || c == '\t' || c == ';') c = ' ';
  }
}

static std::vector<std::string> split_tokens(const std::string& line) {
  std::stringstream ss(line);
  std::vector<std::string> toks;
  while (ss.good()) {
    std::string tok;
    ss >> tok;
    if (!tok.empty()) toks.push_back(tok);
  }
  return toks;
}

std::vector<std::vector<int>> load_workload_tokens_to_ids(
    const std::string& csv_path,
    const std::unordered_map<std::string, int>& vocab,
    bool drop_unknown_tokens,
    bool drop_empty_queries,
    QueryStat* optional_stat
) {
  std::ifstream in(csv_path);
  if (!in) throw std::runtime_error("Cannot open workload file: " + csv_path);

  QueryStat local;
  std::vector<std::vector<int>> queries;

  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;

    normalize_separators(line);

    // skip header line like "Title"
    if (line == "Title" || line == "title" || line == "TITLE") continue;

    auto toks = split_tokens(line);
    if (toks.empty()) continue;

    std::vector<int> q;
    q.reserve(toks.size());

    for (const auto& t : toks) {
      auto it = vocab.find(t);
      if (it == vocab.end()) {
        local.unknown_tokens++;
        continue;
      }
      q.push_back(it->second);
    }

    if (q.empty()) {
      if (drop_empty_queries) {
        local.skipped_queries++;
        continue;
      }
    }

    queries.push_back(std::move(q));
  }

  if (optional_stat) *optional_stat = local;
  return queries;
}

QueryStat benchmark_workload(
    const LBTree& tree,
    const std::vector<std::vector<int>>& queries,
    int warmup
) {
  QueryStat st_warm;
  st_warm.num_queries = (int64_t)queries.size();
  if (st_warm.num_queries == 0) return st_warm;

  // warmup
  for (int i = 0; i < warmup; ++i) {
    for (const auto& q : queries) (void)tree.query(q, &st_warm);
  }

  QueryStat st;
  int64_t total_ns = 0;
  int64_t total_results = 0;
  st.num_queries = (int64_t)queries.size();
  st.internal_ns = 0;
  st.leaf_ns = 0;

  for (const auto& q : queries) {
    auto t0 = std::chrono::steady_clock::now();
    auto ans = tree.query(q, &st);
    auto t1 = std::chrono::steady_clock::now();
    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_results += (int64_t)ans.size();
  }

  st.total_ns = total_ns;
  st.total_results = total_results;
  st.avg_ns = (double)total_ns / (double)st.num_queries;
  st.avg_inter_ns = (double) st.internal_ns / (double) st.num_queries;
  st.avg_leaf_ns = (double) st.leaf_ns / (double) st.num_queries;
  return st;
}
