#pragma once
#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include "Node.h"
#include "BitmapIndexBaseline.h"

class LBTree {
public:

  struct IndexSizeBreakdown {
    uint64_t leaf_bits = 0;      // 所有叶节点总bit数
    uint64_t internal_bits = 0;  // 所有非叶节点总bit数

    uint64_t totalBits() const {
      return leaf_bits + internal_bits;
    }

    uint64_t leafBytes() const {
      return (leaf_bits + 7) / 8;
    }

    uint64_t internalBytes() const {
      return (internal_bits + 7) / 8;
    }

    uint64_t totalBytes() const {
      return (totalBits() + 7) / 8;
    }
  };


  LBTree() = default;

  void buildFromPythonTree(const std::string& outDir);

  std::vector<int64_t> query(const std::vector<int>& keywords, QueryStat* st) const {
    if (!root_) return {};
    return root_->query(keywords, st);
  }

  std::shared_ptr<Node> root() const { return root_; }

  void printSummary() const {
    if (!root_) {
      std::cout << "(empty tree)\n";
      return;
    }
    root_->printSummary(0);
  }

  // ===== index size by paper storage model =====
  IndexSizeBreakdown estimateIndexSizeBreakdown() const;

  uint64_t estimateLeafIndexSizeBits() const;
  uint64_t estimateInternalIndexSizeBits() const;

  uint64_t estimateLeafIndexSizeBytes() const;
  uint64_t estimateInternalIndexSizeBytes() const;

  uint64_t estimateIndexSizeBits() const;
  uint64_t estimateIndexSizeBytes() const;


private:
  std::shared_ptr<Node> root_{nullptr};

  static uint64_t storageBitsLeaf(size_t omega, size_t O) {
    return (uint64_t)omega * (uint64_t)O + 32ULL * (uint64_t)omega + 32ULL * (uint64_t)O;
  }

  static uint64_t storageBitsInternal(size_t omega, size_t C) {
    return (uint64_t)omega * (uint64_t)C + 32ULL * (uint64_t)omega + 32ULL * (uint64_t)C;
  }

  void estimateNodeBitsDFS(const std::shared_ptr<Node>& n,
                           IndexSizeBreakdown& stat) const;
};
