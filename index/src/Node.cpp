#include "Node.h"
#include <chrono>

std::vector<int64_t> Node::query(const std::vector<int>& keywords, QueryStat* st) const {
  std::vector<int64_t> out;
  if (!keywords.empty()) queryInto(keywords, st, out);
  return out;
}

void Node::queryInto(const std::vector<int>& keywords, QueryStat* st,
                     std::vector<int64_t>& out) const {
  using Clock = std::chrono::steady_clock;
  auto start = st ? Clock::now() : Clock::time_point{};
  const bool leaf = isLeaf();
  auto account = [&]() {
    if (st) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
      if (leaf) st->leaf_ns += elapsed;
      else st->internal_ns += elapsed;
    }
  };
  const auto& index = leaf ? kw_obj_bits_ : kw_child_mask_;
  const std::vector<uint64_t>* local[8];
  std::vector<const std::vector<uint64_t>*> overflow;
  if (keywords.size() > 8) overflow.resize(keywords.size());
  auto rows = keywords.size() <= 8 ? local : overflow.data();
  for (size_t k=0; k<keywords.size(); ++k) {
    auto it = index.find(keywords[k]);
    if (it == index.end()) { account(); return; }
    rows[k] = &it->second;
  }
  const size_t limit = leaf ? obj_ids_.size() : children_.size();
  for (size_t w=0; w<rows[0]->size(); ++w) {
    uint64_t x = (*rows[0])[w];
    for (size_t k=1; x && k<keywords.size(); ++k) x &= (*rows[k])[w];
    while (x) {
      const size_t i = (w<<6) + static_cast<unsigned>(__builtin_ctzll(x));
      x &= x-1;
      if (i >= limit) continue;
      if (leaf) out.push_back(obj_ids_[i]);
      else if (children_[i]) {
        account();
        children_[i]->queryInto(keywords, st, out);
        if (st) start = Clock::now();
      }
    }
  }
  account();
}
