#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <optional>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"

class Vocabulary {
public:
  // 从 tree_out/vocab.json 加载：token -> id
  void loadFromJson(const std::string& vocabJsonPath) {
    std::ifstream in(vocabJsonPath);
    if (!in) throw std::runtime_error("Cannot open vocab.json: " + vocabJsonPath);

    nlohmann::json j;
    in >> j;
    if (!j.is_object()) throw std::runtime_error("vocab.json must be a JSON object");

    token2id_.clear();
    id2token_.clear();

    // 先读 token->id
    int maxId = -1;
    for (auto it = j.begin(); it != j.end(); ++it) {
      const std::string token = it.key();
      int id = it.value().get<int>();
      token2id_[token] = id;
      if (id > maxId) maxId = id;
    }

    // 建 id->token（用 vector 更快）
    id2token_.assign((size_t)maxId + 1, "");
    for (const auto& kv : token2id_) {
      int id = kv.second;
      if (id >= 0 && id < (int)id2token_.size()) {
        id2token_[(size_t)id] = kv.first;
      }
    }
  }

  // token -> id（不存在返回 nullopt）
  std::optional<int> idOf(const std::string& token) const {
    auto it = token2id_.find(token);
    if (it == token2id_.end()) return std::nullopt;
    return it->second;
  }

  // id -> token（越界或空返回 "<UNK>"）
  std::string tokenOf(int id) const {
    if (id < 0 || (size_t)id >= id2token_.size()) return "<UNK>";
    const auto& s = id2token_[(size_t)id];
    return s.empty() ? "<UNK>" : s;
  }

  // 给一组 keyword ids，返回对应 token 列表
  std::vector<std::string> tokensOf(const std::vector<int>& ids) const {
    std::vector<std::string> out;
    out.reserve(ids.size());
    for (int id : ids) out.push_back(tokenOf(id));
    return out;
  }

  // 给一组 token，返回 ids（不存在的丢弃或替换为 -1，看你需求）
  std::vector<int> idsOf(const std::vector<std::string>& tokens, bool dropUnknown=true) const {
    std::vector<int> out;
    for (const auto& t : tokens) {
      auto id = idOf(t);
      if (id.has_value()) out.push_back(*id);
      else if (!dropUnknown) out.push_back(-1);
    }
    return out;
  }

  size_t size() const { return token2id_.size(); }

private:
  std::unordered_map<std::string, int> token2id_;
  std::vector<std::string> id2token_;
};
