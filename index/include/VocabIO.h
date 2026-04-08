#pragma once
#include <string>
#include <unordered_map>

std::unordered_map<std::string, int> load_vocab_json(const std::string& vocab_json_path);
