#include "Node.h"
#include <chrono>

static inline bool bitmap_test(const std::vector<uint64_t>& bm, size_t pos) {
  const size_t w = pos >> 6;
  const size_t b = pos & 63;
  return (w < bm.size()) && (((bm[w] >> b) & 1ULL) != 0);
}

std::vector<int64_t> Node::query(const std::vector<int>& keywords, QueryStat* st) const {
  using clock = std::chrono::steady_clock;
  auto ns = [](clock::duration d) -> int64_t {
    return (int64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
  };

  if (keywords.empty()) return {};

  // -------------------- INTERNAL NODE --------------------
  if (!isLeaf()) {
    // seg_start covers "this node's own work" segments (EXCLUDING child subcalls)
    clock::time_point seg_start = clock::now();

    auto internal_add = [&](clock::time_point t) {
      if (st) st->internal_ns += ns(t - seg_start);
      seg_start = t;
    };

    const size_t C = children_.size();

    // include find + mask copy time in internal
    auto it0 = kw_child_mask_.find(keywords[0]);
    if (it0 == kw_child_mask_.end()) {
      internal_add(clock::now());
      return {};
    }

    std::vector<uint64_t> mask = it0->second;

    for (size_t i = 1; i < keywords.size(); ++i) {
      auto it = kw_child_mask_.find(keywords[i]);
      if (it == kw_child_mask_.end()) {
        internal_add(clock::now());
        return {};
      }
      const auto& bm = it->second;

      uint64_t or_acc = 0;
      for (size_t w = 0; w < mask.size(); ++w) {
        mask[w] &= bm[w];
        or_acc |= mask[w];
      }
      if (or_acc == 0) {
        internal_add(clock::now());
        return {};
      }
    }

    std::vector<int64_t> out;
    out.reserve(64);

    for (size_t w = 0; w < mask.size(); ++w) {
      uint64_t x = mask[w];
      while (x) {
        unsigned tz = (unsigned)__builtin_ctzll(x);
        size_t i = (w << 6) + tz;

        if (i < C && children_[i]) {
          // account THIS node's work up to right before recursion
          clock::time_point t_before = clock::now();
          internal_add(t_before);

          // child time is counted inside child (internal/leaf), NOT here
          auto r = children_[i]->query(keywords, st);

          // restart segment AFTER recursion so child time is excluded
          seg_start = clock::now();

          // out.insert belongs to THIS node's internal time
          out.insert(out.end(), r.begin(), r.end());
        }

        x &= (x - 1);
      }
    }

    // account trailing internal segment (including last inserts, loop tail, etc.)
    internal_add(clock::now());
    return out;
  }

  // -------------------- LEAF NODE --------------------
  clock::time_point leaf_start = clock::now();
  auto leaf_finish = [&]() {
    if (st) st->leaf_ns += ns(clock::now() - leaf_start);
  };

  const size_t O = obj_ids_.size();

  auto it0 = kw_obj_bits_.find(keywords[0]);
  if (it0 == kw_obj_bits_.end()) {
    leaf_finish();
    return {};
  }

  std::vector<uint64_t> hit = it0->second;

  for (size_t k = 1; k < keywords.size(); ++k) {
    auto it = kw_obj_bits_.find(keywords[k]);
    if (it == kw_obj_bits_.end()) {
      leaf_finish();
      return {};
    }

    const auto& bm = it->second;

    uint64_t or_acc = 0;
    for (size_t w = 0; w < hit.size(); ++w) {
      hit[w] &= bm[w];
      or_acc |= hit[w];
    }
    if (or_acc == 0) {
      leaf_finish();
      return {};
    }
  }

  std::vector<int64_t> out;
  out.reserve(64);

  for (size_t w = 0; w < hit.size(); ++w) {
    uint64_t x = hit[w];
    while (x) {
      unsigned tz = (unsigned)__builtin_ctzll(x);
      size_t local_idx = (w << 6) + tz;
      if (local_idx < O) out.push_back(obj_ids_[local_idx]);
      x &= (x - 1);
    }
  }

  leaf_finish();
  return out;
}
