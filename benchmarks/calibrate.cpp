// Collect measured bitmap AND / result-enumeration latency for offline Mq fitting.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
int main(int argc, char** argv) {
  if(argc < 2 || argc > 4) {
    std::cerr << "Usage: lbtree_calibrate OUTPUT_CSV [SAMPLES=100000] [MAX_OBJECTS=65536]\n";
    return 1;
  }
  const int samples=argc>2?std::stoi(argv[2]):100000;
  const int max_n=argc>3?std::stoi(argv[3]):65536;
  if(samples<10 || max_n<2) throw std::runtime_error("Invalid calibration dimensions");
  std::ofstream out(argv[1]); if(!out) throw std::runtime_error("Cannot write samples");
  out << "num_objects,query_keyword_count,result_count,query_time_ns\n";
  std::mt19937 rng(42); volatile uint64_t checksum=0;
  for(int s=0;s<samples;++s) {
    const int n=1+int(rng()%max_n),u=1+int(rng()%10),r=int(rng()%(n+1));
    const size_t words=(n+63)/64;
    std::vector<std::vector<uint64_t>> rows(u,std::vector<uint64_t>(words));
    for(int j=0;j<r;++j) for(auto& row:rows) row[j/64]|=uint64_t(1)<<(j%64);
    // The first row limits the intersection to exactly r identifiers.
    for(int q=1;q<u;++q) for(size_t j=0;j<words;++j) rows[q][j]|=uint64_t(rng())<<32|rng();
    std::vector<int> result; result.reserve(r);
    auto evaluate=[&]() {
      result.clear();
      for(size_t w=0;w<words;++w) {
        uint64_t bits=rows[0][w];
        for(int q=1;q<u;++q) bits&=rows[q][w];
        while(bits) { const auto bit=__builtin_ctzll(bits); result.push_back(int(w*64+bit)); bits&=bits-1; }
      }
      checksum+=result.size();
    };
    evaluate(); const auto begin=std::chrono::steady_clock::now();
    for(int repeat=0;repeat<5;++repeat) evaluate();
    const auto elapsed=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-begin).count()/5;
    out << n << ',' << u << ',' << r << ',' << std::max(1.,elapsed) << '\n';
  }
  std::cerr << "Measured samples: " << samples << "; checksum=" << checksum << '\n';
}
