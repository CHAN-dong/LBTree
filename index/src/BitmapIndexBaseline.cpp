#include "BitmapIndexBaseline.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <chrono>

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

static inline bool bitmap_test(const std::vector<uint64_t>& bm, size_t pos) {
  const size_t w = pos >> 6;
  const size_t b = pos & 63;
  return (w < bm.size()) && (((bm[w] >> b) & 1ULL) != 0);
}

static inline void bitmap_set(std::vector<uint64_t>& bm, size_t pos) {
  const size_t w = pos >> 6;
  const size_t b = pos & 63;
  bm[w] |= (uint64_t(1) << b);
}

static inline bool bitmap_any(const std::vector<uint64_t>& bm) {
  for (uint64_t x : bm) {
    if (x) return true;
  }
  return false;
}

void BitmapIndexBaseline::buildFromDatasetCSV(
    const std::string& dataset_csv,
    const std::unordered_map<std::string, int>& vocab,
    bool drop_unknown_tokens
) {
  std::ifstream in(dataset_csv);
  if (!in) throw std::runtime_error("Cannot open dataset file: " + dataset_csv);
  total_object_keyword_occurrences_ = 0;

  std::vector<std::vector<int>> objects;
  objects.reserve(1024);

  std::string line;
  while (std::getline(in, line)) {
    line = trim(line);
    if (line.empty()) continue;

    normalize_separators(line);

    // skip header
    if (line == "Title" || line == "title" || line == "TITLE") continue;

    auto toks = split_tokens(line);
    if (toks.empty()) continue;

    std::vector<int> kw_ids;
    kw_ids.reserve(toks.size());

    for (const auto& t : toks) {
      auto it = vocab.find(t);
      if (it == vocab.end()) {
        if (!drop_unknown_tokens) {
          throw std::runtime_error("Unknown token in dataset: " + t);
        }
        continue;
      }
      kw_ids.push_back(it->second);
    }

    if (!kw_ids.empty()) {
      std::sort(kw_ids.begin(), kw_ids.end());
      kw_ids.erase(std::unique(kw_ids.begin(), kw_ids.end()), kw_ids.end());
      total_object_keyword_occurrences_ += (int64_t)kw_ids.size();  // NEW
      objects.push_back(std::move(kw_ids));
    } else {
      objects.push_back({});
    }
  }


  num_objects_ = (int64_t)objects.size();
  num_words_ = ((size_t)num_objects_ + 63) >> 6;

  bitmaps_.clear();
  bitmaps_.reserve(1024);

  for (int64_t oid = 0; oid < num_objects_; ++oid) {
    for (int kw : objects[(size_t)oid]) {
      auto it = bitmaps_.find(kw);
      if (it == bitmaps_.end()) {
        std::vector<uint64_t> bm(num_words_, 0ULL);
        bitmap_set(bm, (size_t)oid);
        bitmaps_.emplace(kw, std::move(bm));
      } else {
        bitmap_set(it->second, (size_t)oid);
      }
    }
  }
}

std::vector<int64_t> BitmapIndexBaseline::query(const std::vector<int>& keyword_ids) const {
  if (keyword_ids.empty() || num_objects_ == 0) return {};

  auto it0 = bitmaps_.find(keyword_ids[0]);
  if (it0 == bitmaps_.end()) return {};

  std::vector<uint64_t> hit = it0->second;

  for (size_t i = 1; i < keyword_ids.size(); ++i) {
    auto it = bitmaps_.find(keyword_ids[i]);
    if (it == bitmaps_.end()) return {};

    const auto& bm = it->second;
    uint64_t or_acc = 0;
    for (size_t w = 0; w < hit.size(); ++w) {
      hit[w] &= bm[w];
      or_acc |= bm[w];
    }

    if (or_acc == 0) return {};
  }

  std::vector<int64_t> out;
  out.reserve(64);

  for (size_t w = 0; w < hit.size(); ++w) {
    uint64_t x = hit[w];
    while (x) {

      unsigned tz = (unsigned)__builtin_ctzll(x);
      int64_t oid = (int64_t)((w << 6) + tz);

      if (oid < num_objects_) out.push_back(oid);

      x &= (x - 1);
    }
  }

  return out;

}

BaselineBenchStat BitmapIndexBaseline::benchmark(
    const std::vector<std::vector<int>>& queries,
    int warmup_rounds
) const {
  BaselineBenchStat st;
  st.num_queries = (int64_t)queries.size();
  if (st.num_queries == 0) return st;

  // warmup
  for (int w = 0; w < warmup_rounds; ++w) {
    for (const auto& q : queries) {
      (void)this->query(q);
    }
  }

  int64_t total_ns = 0;
  int64_t total_results = 0;

  for (const auto& q : queries) {
    auto t0 = std::chrono::steady_clock::now();
    auto ans = this->query(q);
    auto t1 = std::chrono::steady_clock::now();

    total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    total_results += (int64_t)ans.size();
  }

  st.total_ns = total_ns;
  st.total_results = total_results;
  st.avg_ns = (double)total_ns / (double)st.num_queries;
  return st;
}

void BitmapIndexBaseline::exportSamplesCSVBasic(
    const std::vector<std::vector<int>>& queries,
    const std::string& out_csv,
    int warmup_rounds,
    int repeat_per_query
) const {
  if (repeat_per_query <= 0) repeat_per_query = 1;

  std::ofstream out(out_csv);
  if (!out) throw std::runtime_error("Cannot open output csv file: " + out_csv);

  out << "num_objects,"
      << "dataset_total_keyword_occurrences,"
      << "dataset_distinct_keywords,"
      << "query_keyword_count,"
      << "result_count,"
      << "query_time_ns\n";

  // warmup
  for (int w = 0; w < warmup_rounds; ++w) {
    for (const auto& q : queries) {
      (void)this->query(q);
    }
  }

  for (const auto& q : queries) {
    const int query_keyword_count = (int)q.size();

    for (int r = 0; r < repeat_per_query; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      auto ans = this->query(q);
      auto t1 = std::chrono::steady_clock::now();

      int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

      out << num_objects_ << ","
          << total_object_keyword_occurrences_ << ","
          << (int64_t)bitmaps_.size() << ","
          << query_keyword_count << ","
          << (int64_t)ans.size() << ","
          << ns << "\n";
    }
  }
}

void BitmapIndexBaseline::buildFromObjects(const std::vector<std::vector<int>>& objects) {
  total_object_keyword_occurrences_ = 0;

  num_objects_ = (int64_t)objects.size();
  num_words_ = ((size_t)num_objects_ + 63) >> 6;

  bitmaps_.clear();
  bitmaps_.reserve(1024);

  for (int64_t oid = 0; oid < num_objects_; ++oid) {
    std::vector<int> kw_ids = objects[(size_t)oid];
    std::sort(kw_ids.begin(), kw_ids.end());
    kw_ids.erase(std::unique(kw_ids.begin(), kw_ids.end()), kw_ids.end());

    total_object_keyword_occurrences_ += (int64_t)kw_ids.size();

    for (int kw : kw_ids) {
      auto it = bitmaps_.find(kw);
      if (it == bitmaps_.end()) {
        std::vector<uint64_t> bm(num_words_, 0ULL);
        const size_t w = ((size_t)oid) >> 6;
        const size_t b = ((size_t)oid) & 63;
        bm[w] |= (uint64_t(1) << b);
        bitmaps_.emplace(kw, std::move(bm));
      } else {
        const size_t w = ((size_t)oid) >> 6;
        const size_t b = ((size_t)oid) & 63;
        it->second[w] |= (uint64_t(1) << b);
      }
    }
  }
}