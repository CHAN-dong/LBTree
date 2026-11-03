#include "LBTree.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

using Clock = std::chrono::steady_clock;
template<class T> void put(std::ofstream& f, T v) { f.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
int main(int argc, char** argv) {
  const int N = argc > 1 ? std::stoi(argv[1]) : 4096;
  if (N <= 0) throw std::runtime_error("N must be positive");
  const int V=128, L=65, C=(N+L-1)/L;
  std::mt19937 rng(42);
  for (size_t bits=0;bits<140;++bits) {
    std::vector<uint8_t> bytes((bits+7)/8+2);
    for(auto& b:bytes) b=static_cast<uint8_t>(rng());
    for(size_t available=0;available<=bytes.size();++available) {
      auto actual=BitmapIndexBaseline::FromPackedLittle(bytes.data(),available,bits);
      std::vector<uint64_t> expected((bits+63)/64);
      for(size_t b=0;b<bits;++b) if(b/8<available && ((bytes[b/8]>>(b%8))&1)) expected[b/64]|=uint64_t(1)<<(b%64);
      if(actual!=expected) throw std::runtime_error("packed conversion mismatch");
    }
  }
  rng.seed(42);
  std::vector<std::vector<unsigned char>> objects(N, std::vector<unsigned char>(V));
  for(auto& o: objects) for(int k=0;k<12;++k) o[rng()%V]=1;
  std::filesystem::create_directories("benchmarks/data");
  std::ofstream f("benchmarks/data/tree.bin", std::ios::binary);
  f.write("LBTREE1\0",8); put<int32_t>(f,1); put<int32_t>(f,0); put<int32_t>(f,C+1);
  put<int32_t>(f,0); put<uint8_t>(f,0); put<int32_t>(f,C);
  for(int c=0;c<C;++c) put<int32_t>(f,c+1);
  put<int32_t>(f,V);
  for(int k=0;k<V;++k) {
    put<int32_t>(f,k); std::vector<uint8_t> b((C+7)/8);
    for(int c=0;c<C;++c) for(int o=c*L;o<std::min(N,(c+1)*L);++o) if(objects[o][k]) { b[c/8]|=1u<<(c%8); break; }
    f.write(reinterpret_cast<char*>(b.data()),b.size());
  }
  for(int c=0;c<C;++c) {
    int count=std::min(L,N-c*L);
    put<int32_t>(f,c+1); put<uint8_t>(f,1); put<int32_t>(f,count);
    for(int o=0;o<count;++o) put<int64_t>(f,c*L+o);
    put<int32_t>(f,V);
    for(int k=0;k<V;++k) {
      put<int32_t>(f,k); std::vector<uint8_t> b((count+7)/8);
      for(int o=0;o<count;++o) if(objects[c*L+o][k]) b[o/8]|=1u<<(o%8);
      f.write(reinterpret_cast<char*>(b.data()),b.size());
    }
  }
  f.close();
  LBTree tree; std::vector<double> builds;
  for(int r=0;r<9;++r) { auto t=Clock::now(); tree.buildFromPythonTree("benchmarks/data"); builds.push_back(std::chrono::duration<double,std::milli>(Clock::now()-t).count()); }
  auto nested=std::make_shared<Node>(-1,Node::Type::Internal);
  nested->setChildren({tree.root()});
  for(int k=0;k<V;++k) nested->addKeywordChildMask(k,{1});
  std::vector<std::vector<int>> qs={{},{-1},{0},{0,0},{0,129},std::vector<int>(12,0)};
  for(int i=0;i<1000;++i) { std::vector<int> q; for(int k=0;k<1+i%5;++k) q.push_back(rng()%V); qs.push_back(q); }
  for(const auto& q:qs) {
    std::vector<int64_t> expected;
    if(!q.empty()) for(int o=0;o<N;++o) { bool ok=true; for(int k:q) if(k<0||k>=V||!objects[o][k]) {ok=false;break;} if(ok) expected.push_back(o); }
    auto actual=tree.query(q,nullptr); if(actual!=expected) throw std::runtime_error("query mismatch");
    QueryStat st; if(tree.query(q,&st)!=expected) throw std::runtime_error("instrumented mismatch");
    if(nested->query(q,&st)!=expected) throw std::runtime_error("nested mismatch");
  }
  uint64_t checksum=0; std::vector<double> times;
  for(int r=0;r<9;++r) { auto t=Clock::now(); for(int rep=0;rep<5;++rep) for(auto& q:qs) checksum+=tree.query(q,nullptr).size(); times.push_back(std::chrono::duration<double,std::nano>(Clock::now()-t).count()/(5*qs.size())); }
  std::sort(builds.begin(),builds.end()); std::sort(times.begin(),times.end());
  std::cout<<"objects="<<N<<" queries="<<qs.size()<<" correct=true build_ms="<<builds[4]<<" query_ns="<<times[4]<<" checksum="<<checksum<<"\n";
}
