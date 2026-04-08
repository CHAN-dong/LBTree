#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

struct BaselineBenchStat {
  int64_t num_queries = 0;
  int64_t total_ns = 0;
  double avg_ns = 0.0;
  int64_t total_results = 0;
};

class BitmapIndexBaseline {
public:
  // dataset_csv: 每行一个对象，token 用空格/逗号分隔，第一行可为 Title
  // vocab: token -> id
  void buildFromDatasetCSV(
      const std::string& dataset_csv,
      const std::unordered_map<std::string, int>& vocab,
      bool drop_unknown_tokens = true
  );

  // conjunctive query (AND)
  std::vector<int64_t> query(const std::vector<int>& keyword_ids) const;

  // ===== benchmark baseline on workload queries =====
  BaselineBenchStat benchmark(
      const std::vector<std::vector<int>>& queries,
      int warmup_rounds = 1
  ) const;

  size_t numObjects() const { return (size_t)num_objects_; }
  size_t numIndexedKeywords() const { return bitmaps_.size(); }

  uint64_t estimateIndexSizeBits() const {
    uint64_t V = (uint64_t)numIndexedKeywords();
    uint64_t N = (uint64_t)numObjects();
    return V * N + 32ULL * V + 32ULL * N;
  }

  uint64_t estimateIndexSizeBytes() const {
    uint64_t bits = estimateIndexSizeBits();
    return (bits + 7) / 8;
  }

static std::vector<uint64_t> FromPackedLittle(const uint8_t* bytes, size_t nbytes, size_t numBits) {
  const size_t nwords = (numBits + 63) >> 6;
  std::vector<uint64_t> bm(nwords, 0ULL);
  for (size_t i = 0; i < numBits; ++i) {
    const size_t by = i >> 3;   // byte index
    const size_t bi = i & 7;    // bit index within byte (little-endian)
    if (by >= nbytes) break;
    const bool v = (((bytes[by] >> bi) & 1U) != 0);
    if (v) {
      const size_t w = i >> 6;
      const size_t b = i & 63;
      bm[w] |= (uint64_t(1) << b);
    }
  }
  return bm;
}

// NEW: 导出逐query sample（用于拟合cost model / NN）
void exportSamplesCSVBasic(
    const std::vector<std::vector<int>>& queries,
    const std::string& out_csv,
    int warmup_rounds = 3,
    int repeat_per_query = 5
) const;

void buildFromObjects(const std::vector<std::vector<int>>& objects);

// 可选 getter（训练时也许有用）
int64_t totalObjectKeywordOccurrences() const { return total_object_keyword_occurrences_; }
int64_t distinctKeywords() const { return (int64_t)bitmaps_.size(); }




private:
  int64_t num_objects_ = 0;
  size_t num_words_ = 0; // ceil(num_objects_/64)

  // kw_id -> bitmap over objects (packed uint64_t words)
  std::unordered_map<int, std::vector<uint64_t>> bitmaps_;

 // NEW: 数据集级特征（构建时统计）
  int64_t total_object_keyword_occurrences_ = 0; // sum over objects of unique keyword count

};