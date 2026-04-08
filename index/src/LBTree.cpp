#include "LBTree.h"
#include "NpyReader.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include "nlohmann/json.hpp"

static std::string joinPath(const std::string& dir, const std::string& file) {
  if (dir.empty()) return file;
  if (dir.back() == '/' || dir.back() == '\\') return dir + file;
  return dir + "/" + file;
}

void LBTree::estimateNodeBitsDFS(const std::shared_ptr<Node>& n,
                                 IndexSizeBreakdown& stat) const {
  if (!n) return;

  if (n->isLeaf()) {
    size_t omega = n->keywordCount();   // |Ω_O|
    size_t O = n->objectCount();        // |O|
    stat.leaf_bits += storageBitsLeaf(omega, O);
    return;
  }

  // internal node
  size_t omega = n->keywordCount();  // |Ω_O|
  size_t C = n->childCount();        // #branches
  stat.internal_bits += storageBitsInternal(omega, C);

  for (const auto& ch : n->children()) {
    estimateNodeBitsDFS(ch, stat);
  }
}

LBTree::IndexSizeBreakdown LBTree::estimateIndexSizeBreakdown() const {
  IndexSizeBreakdown stat;
  if (!root_) return stat;
  estimateNodeBitsDFS(root_, stat);
  return stat;
}

uint64_t LBTree::estimateLeafIndexSizeBits() const {
  return estimateIndexSizeBreakdown().leaf_bits;
}

uint64_t LBTree::estimateInternalIndexSizeBits() const {
  return estimateIndexSizeBreakdown().internal_bits;
}

uint64_t LBTree::estimateLeafIndexSizeBytes() const {
  uint64_t bits = estimateLeafIndexSizeBits();
  return (bits + 7) / 8;
}

uint64_t LBTree::estimateInternalIndexSizeBytes() const {
  uint64_t bits = estimateInternalIndexSizeBits();
  return (bits + 7) / 8;
}

uint64_t LBTree::estimateIndexSizeBits() const {
  return estimateIndexSizeBreakdown().totalBits();
}

uint64_t LBTree::estimateIndexSizeBytes() const {
  uint64_t bits = estimateIndexSizeBits();
  return (bits + 7) / 8;
}


// void LBTree::buildFromPythonTree(const std::string& outDir) {
//   using json = nlohmann::json;

//   std::ifstream in(joinPath(outDir, "tree.json"));
//   if (!in) throw std::runtime_error("Cannot open tree.json in " + outDir);
//   json j; in >> j;

//   int rootId = j.at("root").get<int>();
//   auto nodes = j.at("nodes");

//   int maxId = -1;
//   for (auto& nd : nodes) maxId = std::max(maxId, nd.at("node_id").get<int>());

//   std::vector<std::shared_ptr<Node>> ptr(maxId + 1, nullptr);

//   // create nodes
//   for (auto& nd : nodes) {
//     int id = nd.at("node_id").get<int>();
//     std::string kind = nd.at("kind").get<std::string>();
//     ptr[id] = std::make_shared<Node>(id, kind == "leaf" ? Node::Type::Leaf : Node::Type::Internal);
//   }

//   // load payload
//   for (auto& nd : nodes) {
//     int id = nd.at("node_id").get<int>();
//     std::string kind = nd.at("kind").get<std::string>();
//     auto data = nd.at("data");

//     if (kind == "internal") {
//       // NEW FORMAT:
//       // children_ids: int64 [C]
//       // kw_ids: int64 [K]
//       // kw_child_bits: uint8 [K, ceil(C/8)] packed little

//       auto childArr = load_npy<int64_t>(joinPath(outDir, data.at("children_ids").get<std::string>()));
//       auto kwArr    = load_npy<int64_t>(joinPath(outDir, data.at("kw_ids").get<std::string>()));
//       auto matArr   = load_npy<uint8_t>(joinPath(outDir, data.at("kw_child_bits").get<std::string>()));

//       if (childArr.shape.size() != 1) throw std::runtime_error("children_ids must be 1D");
//       size_t C = childArr.data.size();

//       if (kwArr.shape.size() != 1) throw std::runtime_error("kw_ids must be 1D");
//       size_t K = kwArr.data.size();

//       if (matArr.shape.size() != 2 || matArr.shape[0] != K)
//         throw std::runtime_error("kw_child_bits must be 2D with first dim=K");
//       size_t bytesPerRow = matArr.shape[1];
//       if (bytesPerRow != (C + 7) / 8)
//         throw std::runtime_error("kw_child_bits bytes_per_row mismatch");

//       // store masks: kw -> DynamicBitset(C)
//       for (size_t k = 0; k < K; ++k) {
//         int kw = (int)kwArr.data[k];
//         const uint8_t* row = matArr.data.data() + k * bytesPerRow;
//         auto mask = BitmapIndexBaseline::FromPackedLittle(row, bytesPerRow, C);
//         ptr[id]->addKeywordChildMask(kw, mask);
//       }

//     } else {
//       // leaf (same as before)
//       auto objArr = load_npy<int64_t>(joinPath(outDir, data.at("obj_ids").get<std::string>()));
//       auto kwArr  = load_npy<int64_t>(joinPath(outDir, data.at("kw_ids").get<std::string>()));
//       auto matArr = load_npy<uint8_t>(joinPath(outDir, data.at("kw_obj_bits").get<std::string>()));

//       ptr[id]->setLeafObjects(objArr.data);
//       size_t O = objArr.data.size();

//       size_t K = kwArr.data.size();
//       if (K == 0) continue;

//       if (matArr.shape.size() != 2 || matArr.shape[0] != K)
//         throw std::runtime_error("leaf kw_obj_bits must be 2D with first dim=K");
//       size_t bytesPerRow = matArr.shape[1];
//       if (bytesPerRow != (O + 7) / 8)
//         throw std::runtime_error("leaf kw_obj_bits bytes_per_row mismatch");

//       for (size_t k = 0; k < K; ++k) {
//         int kw = (int)kwArr.data[k];
//         const uint8_t* row = matArr.data.data() + k * bytesPerRow;
//         auto bs = BitmapIndexBaseline::FromPackedLittle(row, bytesPerRow, O);
//         ptr[id]->addKeywordObjectBits(kw, bs);
//       }
//     }
//   }

//   // connect children (NEW: vector children)
//   for (auto& nd : nodes) {
//     bool isLeaf = nd.at("is_leaf").get<bool>();
//     if (isLeaf) continue;

//     int id = nd.at("node_id").get<int>();
//     auto data = nd.at("data");

//     auto childArr = load_npy<int64_t>(joinPath(outDir, data.at("children_ids").get<std::string>()));
//     std::vector<std::shared_ptr<Node>> children;
//     children.reserve(childArr.data.size());
//     for (auto cid : childArr.data) {
//       int c = (int)cid;
//       children.push_back(ptr[c]);
//     }
//     ptr[id]->setChildren(std::move(children));
//   }

//   root_ = ptr[rootId];
// }


#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr char kTreeMagic[8] = {'L', 'B', 'T', 'R', 'E', 'E', '1', '\0'};

int32_t readI32(std::istream& in) {
  int32_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) throw std::runtime_error("Failed to read int32 from tree.bin");
  return v;
}

uint8_t readU8(std::istream& in) {
  uint8_t v = 0;
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  if (!in) throw std::runtime_error("Failed to read uint8 from tree.bin");
  return v;
}

std::vector<int32_t> readI32Vec(std::istream& in, size_t n) {
  std::vector<int32_t> v(n);
  if (n > 0) {
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(int32_t)));
    if (!in) throw std::runtime_error("Failed to read int32 array from tree.bin");
  }
  return v;
}

std::vector<int64_t> readI64Vec(std::istream& in, size_t n) {
  std::vector<int64_t> v(n);
  if (n > 0) {
    in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(int64_t)));
    if (!in) throw std::runtime_error("Failed to read int64 array from tree.bin");
  }
  return v;
}

void readBytes(std::istream& in, uint8_t* dst, size_t n) {
  if (n == 0) return;
  in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
  if (!in) throw std::runtime_error("Failed to read raw bytes from tree.bin");
}
}  // namespace


void LBTree::buildFromPythonTree(const std::string& outDir) {
  std::ifstream in(joinPath(outDir, "tree.bin"), std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open tree.bin in " + outDir);

  char magic[8];
  in.read(magic, 8);
  if (!in || std::memcmp(magic, kTreeMagic, 8) != 0) {
    throw std::runtime_error("Invalid tree.bin magic in " + outDir);
  }

  int32_t version = readI32(in);
  if (version != 1) {
    throw std::runtime_error("Unsupported tree.bin version: " + std::to_string(version));
  }

  int32_t rootId = readI32(in);
  int32_t numNodes = readI32(in);
  if (numNodes < 0) {
    throw std::runtime_error("Invalid numNodes in tree.bin");
  }

  std::unordered_map<int, std::shared_ptr<Node>> ptr;
  std::unordered_map<int, std::vector<int>> pendingChildren;
  ptr.reserve(static_cast<size_t>(numNodes) * 2);
  pendingChildren.reserve(static_cast<size_t>(numNodes) * 2);

  for (int32_t ni = 0; ni < numNodes; ++ni) {
    int32_t id = readI32(in);
    bool isLeaf = (readU8(in) != 0);

    auto node = std::make_shared<Node>(
        static_cast<int>(id),
        isLeaf ? Node::Type::Leaf : Node::Type::Internal
    );
    ptr[static_cast<int>(id)] = node;

    if (isLeaf) {
      int32_t O = readI32(in);
      if (O < 0) throw std::runtime_error("Negative obj_count in tree.bin");

      auto objIds = readI64Vec(in, static_cast<size_t>(O));
      node->setLeafObjects(objIds);

      int32_t K = readI32(in);
      if (K < 0) throw std::runtime_error("Negative kw_count in tree.bin");

      size_t bytesPerRow = (static_cast<size_t>(O) + 7) / 8;
      std::vector<uint8_t> row(bytesPerRow);

      for (int32_t k = 0; k < K; ++k) {
        int kw = static_cast<int>(readI32(in));
        readBytes(in, row.data(), bytesPerRow);

        auto bs = BitmapIndexBaseline::FromPackedLittle(
            row.data(), bytesPerRow, static_cast<size_t>(O));
        node->addKeywordObjectBits(kw, bs);
      }

    } else {
      int32_t C = readI32(in);
      if (C < 0) throw std::runtime_error("Negative child_count in tree.bin");

      auto childIds32 = readI32Vec(in, static_cast<size_t>(C));
      std::vector<int> childIds;
      childIds.reserve(static_cast<size_t>(C));
      for (auto cid : childIds32) childIds.push_back(static_cast<int>(cid));
      pendingChildren[static_cast<int>(id)] = std::move(childIds);

      int32_t K = readI32(in);
      if (K < 0) throw std::runtime_error("Negative kw_count in tree.bin");

      size_t bytesPerRow = (static_cast<size_t>(C) + 7) / 8;
      std::vector<uint8_t> row(bytesPerRow);

      for (int32_t k = 0; k < K; ++k) {
        int kw = static_cast<int>(readI32(in));
        readBytes(in, row.data(), bytesPerRow);

        auto mask = BitmapIndexBaseline::FromPackedLittle(
            row.data(), bytesPerRow, static_cast<size_t>(C));
        node->addKeywordChildMask(kw, mask);
      }
    }
  }

  // connect children after all nodes are created
  for (auto& kv : pendingChildren) {
    int id = kv.first;
    const auto& childIds = kv.second;

    std::vector<std::shared_ptr<Node>> children;
    children.reserve(childIds.size());

    for (int cid : childIds) {
      auto it = ptr.find(cid);
      if (it == ptr.end()) {
        throw std::runtime_error("Child node id not found while connecting children: " + std::to_string(cid));
      }
      children.push_back(it->second);
    }
    ptr.at(id)->setChildren(std::move(children));
  }

  auto itRoot = ptr.find(static_cast<int>(rootId));
  if (itRoot == ptr.end()) {
    throw std::runtime_error("Root id not found in tree.bin: " + std::to_string(rootId));
  }
  root_ = itRoot->second;
}