#pragma once
#include <unordered_map>
#include <memory>
#include <vector>
#include <cstdint>
#include <iostream>
#include "WorkloadIO.h"

class Node {
public:
  enum class Type { Internal, Leaf };

  explicit Node(int id, Type t) : id_(id), type_(t) {}

  int id() const { return id_; }
  bool isLeaf() const { return type_ == Type::Leaf; }

  // ===== children (support C>2) =====
  void setChildren(std::vector<std::shared_ptr<Node>> children) {
    children_ = std::move(children);
  }

  size_t childCount() const { return children_.size(); }

  // internal: keyword -> child mask bitset (size = childCount)
  void addKeywordChildMask(int kw, const std::vector<uint64_t>& mask) {
    kw_child_mask_[kw] = mask;
  }

  // leaf objects (global ids)
  void setLeafObjects(std::vector<int64_t> objIds) { obj_ids_ = std::move(objIds); }

  // leaf: keyword -> object bitmap (size = obj_ids_.size())
  void addKeywordObjectBits(int kw, const std::vector<uint64_t>& bits) {
    kw_obj_bits_[kw] = bits;
  }

  std::vector<int64_t> query(const std::vector<int>& keywords, QueryStat* st) const;

  // ===== summary stats you want =====
  size_t keywordCount() const {
    return isLeaf() ? kw_obj_bits_.size() : kw_child_mask_.size();
  }

  size_t objectCount() const {
    return isLeaf() ? obj_ids_.size() : 0;
  }

  void printSummary(int depth = 0) const {
    const char* t = isLeaf() ? "Leaf" : "Internal";
    for (int i = 0; i < depth; ++i) std::cout << "  ";
    std::cout << "- node=" << id_
              << " depth=" << depth
              << " type=" << t
              << " #kw=" << keywordCount()
              << " #obj=" << objectCount()
              << " #child=" << childCount()
              << "\n";
    for (auto& ch : children_) {
      if (ch) ch->printSummary(depth + 1);
    }
  }

  const std::vector<std::shared_ptr<Node>>& children() const { return children_; }

private:
  int id_;
  Type type_;

  // internal
  std::vector<std::shared_ptr<Node>> children_;
  std::unordered_map<int, std::vector<uint64_t>> kw_child_mask_; // bitset size = childCount()

  // leaf
  std::vector<int64_t> obj_ids_;
  std::unordered_map<int, std::vector<uint64_t>> kw_obj_bits_;   // bitset size = |O_leaf|
};
