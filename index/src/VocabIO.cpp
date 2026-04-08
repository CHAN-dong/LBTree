#include "VocabIO.h"
#include <fstream>
#include <stdexcept>
#include "nlohmann/json.hpp"

std::unordered_map<std::string, int> load_vocab_json(const std::string& vocab_json_path) {
  using json = nlohmann::json;

  std::ifstream in(vocab_json_path);
  if (!in) throw std::runtime_error("Cannot open vocab.json: " + vocab_json_path);

  json j;
  in >> j;

  std::unordered_map<std::string, int> vocab;
  vocab.reserve(j.size());

  for (auto it = j.begin(); it != j.end(); ++it) {
    vocab[it.key()] = it.value().get<int>();
  }
  return vocab;
}
