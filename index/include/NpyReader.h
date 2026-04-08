#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <stdexcept>
#include <numeric>

struct NpyHeader {
  std::string descr;
  bool fortran_order = false;
  std::vector<size_t> shape;
};

inline std::string read_exact(std::ifstream& in, size_t n) {
  std::string s(n, '\0');
  in.read(&s[0], (std::streamsize)n);
  if ((size_t)in.gcount() != n) throw std::runtime_error("NPY read_exact failed");
  return s;
}

inline NpyHeader parse_header(const std::string& hdr) {
  // very small parser for python dict-like header:
  // {'descr': '<i8', 'fortran_order': False, 'shape': (10,), }
  NpyHeader h;

  auto find_str = [&](const std::string& key)->std::string{
    auto pos = hdr.find(key);
    if (pos == std::string::npos) throw std::runtime_error("NPY header missing key: " + key);
    pos = hdr.find("'", pos);
    pos = hdr.find("'", pos+1);
    auto start = pos+1;
    auto end = hdr.find("'", start);
    return hdr.substr(start, end-start);
  };

  // descr
  {
    auto p = hdr.find("'descr'");
    if (p == std::string::npos) throw std::runtime_error("NPY header missing descr");
    auto q = hdr.find(":", p);
    auto s = hdr.find("'", q);
    auto e = hdr.find("'", s+1);
    h.descr = hdr.substr(s+1, e-s-1);
  }

  // fortran_order
  {
    auto p = hdr.find("'fortran_order'");
    if (p == std::string::npos) throw std::runtime_error("NPY header missing fortran_order");
    auto q = hdr.find(":", p);
    auto t = hdr.substr(q+1);
    h.fortran_order = (t.find("True") != std::string::npos);
  }

  // shape
  {
    auto p = hdr.find("'shape'");
    if (p == std::string::npos) throw std::runtime_error("NPY header missing shape");
    auto q = hdr.find("(", p);
    auto r = hdr.find(")", q);
    auto inside = hdr.substr(q+1, r-q-1);

    std::vector<size_t> shp;
    std::stringstream ss(inside);
    while (ss.good()) {
      std::string tok;
      std::getline(ss, tok, ',');
      // trim spaces
      while (!tok.empty() && tok.front()==' ') tok.erase(tok.begin());
      while (!tok.empty() && tok.back()==' ') tok.pop_back();
      if (tok.empty()) continue;
      shp.push_back((size_t)std::stoull(tok));
    }
    h.shape = shp;
  }

  return h;
}

inline NpyHeader read_npy_header(std::ifstream& in) {
  auto magic = read_exact(in, 6);
  if (magic != "\x93NUMPY") throw std::runtime_error("Not a NPY file");

  uint8_t major = 0, minor = 0;
  in.read((char*)&major, 1);
  in.read((char*)&minor, 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t hl16 = 0;
    in.read((char*)&hl16, 2);
    header_len = hl16;
  } else if (major == 2) {
    uint32_t hl32 = 0;
    in.read((char*)&hl32, 4);
    header_len = hl32;
  } else {
    throw std::runtime_error("Unsupported NPY version");
  }

  auto hdr = read_exact(in, header_len);
  auto h = parse_header(hdr);
  if (h.fortran_order) throw std::runtime_error("Fortran order NPY not supported (need C-order)");
  return h;
}

template<typename T>
struct NpyArray {
  std::vector<size_t> shape;
  std::vector<T> data;
};

inline size_t numel(const std::vector<size_t>& shape) {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), (size_t)1, [](size_t a, size_t b){ return a*b; });
}

template<typename T>
inline NpyArray<T> load_npy(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open npy: " + path);

  auto h = read_npy_header(in);

  // validate dtype
  if constexpr (std::is_same<T, int64_t>::value) {
    if (h.descr != "<i8" && h.descr != "|i8") throw std::runtime_error("NPY dtype mismatch (expect int64): " + h.descr);
  } else if constexpr (std::is_same<T, uint8_t>::value) {
    if (h.descr != "|u1" && h.descr != "<u1") throw std::runtime_error("NPY dtype mismatch (expect uint8): " + h.descr);
  } else {
    throw std::runtime_error("Unsupported template dtype for NPY");
  }

  size_t n = numel(h.shape);
  NpyArray<T> arr;
  arr.shape = h.shape;
  arr.data.resize(n);

  in.read((char*)arr.data.data(), (std::streamsize)(n * sizeof(T)));
  if (!in) throw std::runtime_error("NPY data read failed: " + path);
  return arr;
}

