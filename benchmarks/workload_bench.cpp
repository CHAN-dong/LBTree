#include "LBTree.h"
#include "VocabIO.h"
#include "WorkloadIO.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <cstring>
#include "nlohmann/json.hpp"

using Clock=std::chrono::steady_clock;
using json=nlohmann::json;
int warmup_rounds=10, measurement_rounds=30;
double ns(Clock::duration d) { return std::chrono::duration<double,std::nano>(d).count(); }
double quantile(std::vector<double> x,double p) { std::sort(x.begin(),x.end()); return x[static_cast<size_t>(p*(x.size()-1))]; }
template<class Index> json measure(const Index& index,const std::vector<std::vector<int>>& queries) {
  uint64_t checksum=0;
  for(int r=0;r<warmup_rounds;++r) for(auto& q:queries) checksum+=index.query(q).size();
  std::vector<size_t> order(queries.size()); std::iota(order.begin(),order.end(),0);
  std::mt19937 rng(42);
  std::vector<double> samples,round_averages;
  // Keep result materialization and destruction inside the timed query call.
  for(int round=0;round<measurement_rounds;++round) {
    std::shuffle(order.begin(),order.end(),rng);
    double total=0;
    for(auto i:order) {
      auto t=Clock::now();
      { auto result=index.query(queries[i]); checksum+=result.size(); }
      double dt=ns(Clock::now()-t); samples.push_back(dt); total+=dt;
    }
    round_averages.push_back(total/queries.size());
  }
  return {{"mean_ns",std::accumulate(samples.begin(),samples.end(),0.0)/samples.size()},
          {"p50_ns",quantile(samples,.50)},{"p95_ns",quantile(samples,.95)},{"p99_ns",quantile(samples,.99)},
          {"median_round_mean_ns",quantile(round_averages,.5)},
          {"measured_calls",samples.size()},{"checksum_including_warmup",checksum}};
}
struct TreeView {
  const LBTree& tree;
  std::vector<int64_t> query(const std::vector<int>& q) const { return tree.query(q,nullptr); }
};
int main(int argc,char** argv) {
  const bool use_expected=argc>=3 && std::string(argv[2])=="--expected";
  const bool split_stats=argc==4 && std::string(argv[3])=="--split-stats";
  if(argc!=2 && !use_expected) { std::cerr<<"Usage: lbtree_workload_bench EXPERIMENT_DIRECTORY [--expected]\n";return 1; }
  if(use_expected) {warmup_rounds=2;measurement_rounds=5;}
  std::filesystem::path directory=argv[1];
  auto vocab=load_vocab_json((directory/"vocab.json").string());
  QueryStat io;
  auto queries=load_workload_tokens_to_ids((directory/"workload.csv").string(),vocab,false,false,&io);
  if(queries.empty()||io.unknown_tokens) throw std::runtime_error("Missing queries or vocabulary mismatch");
  json report;
  report["queries"]=queries.size(); report["warmup_rounds"]=warmup_rounds; report["measurement_rounds"]=measurement_rounds;
  report["timer_includes"]="AND bitmap traversal, result materialization and destruction; excludes parsing, IO and per-node timers";
  std::vector<double> clock_samples;
  for(int i=0;i<10000;++i) { auto t=Clock::now();clock_samples.push_back(ns(Clock::now()-t)); }
  report["clock_pair_median_ns"]=quantile(clock_samples,.5);
  BitmapIndexBaseline baseline;
  auto t=Clock::now();
  std::vector<std::vector<int64_t>> expected;
  if(use_expected) {
    std::ifstream in(directory/"expected.bin",std::ios::binary);char magic[8];uint64_t count=0;
    in.read(magic,8);in.read(reinterpret_cast<char*>(&count),8);
    if(!in||std::memcmp(magic,"LBANS1\0\0",8)||count!=queries.size()) throw std::runtime_error("Invalid exact-answer file");
    expected.resize(count);
    for(auto& row:expected) {
      uint64_t size=0;in.read(reinterpret_cast<char*>(&size),8);row.resize(size);
      in.read(reinterpret_cast<char*>(row.data()),static_cast<std::streamsize>(size*8));
      if(!in) throw std::runtime_error("Truncated exact-answer file");
    }
    std::ifstream mf(directory/"metadata.json");json meta;mf>>meta;report["objects"]=meta.at("objects");
    report["verification_reference"]="exact conjunctive postings intersections, no flat bitmap allocation";
  } else {
    baseline.buildFromDatasetCSV((directory/"objects.csv").string(),vocab,false);
    report["bitmap_baseline_build_seconds"]=ns(Clock::now()-t)/1e9;
    report["objects"]=baseline.numObjects();
    for(auto& q:queries) expected.push_back(baseline.query(q));
  }
  for(auto kind:{"binary","fanout"}) {
    const auto path=directory/kind;
    LBTree tree; std::vector<double> load;
    const int loads=std::filesystem::file_size(path/"tree.bin")>uint64_t(1024)*1024*1024 ? 1:5;
    for(int i=0;i<loads;++i) { LBTree fresh; t=Clock::now(); fresh.buildFromPythonTree(path.string());
      load.push_back(ns(Clock::now()-t)/1e6); tree=std::move(fresh); }
    uint64_t result_count=0,nonempty=0;QueryStat split_timing;
    for(size_t i=0;i<queries.size();++i) {
      auto actual=tree.query(queries[i],split_stats?&split_timing:nullptr); std::sort(actual.begin(),actual.end());
      if(actual!=expected[i]) throw std::runtime_error(std::string(kind)+" result mismatch at query "+std::to_string(i));
      result_count+=actual.size(); nonempty+=!actual.empty();
    }
    auto stats=measure(TreeView{tree},queries);
    if(split_stats){stats["instrumented_internal_mean_ns"]=double(split_timing.internal_ns)/queries.size();stats["instrumented_leaf_mean_ns"]=double(split_timing.leaf_ns)/queries.size();}
    std::vector<std::vector<int>> hit_queries,empty_queries;
    for(size_t i=0;i<queries.size();++i)
      (expected[i].empty()?empty_queries:hit_queries).push_back(queries[i]);
    if(!hit_queries.empty()) stats["nonempty_query_latency"]=measure(TreeView{tree},hit_queries);
    if(!empty_queries.empty()) stats["empty_query_latency"]=measure(TreeView{tree},empty_queries);
    auto size=tree.estimateIndexSizeBreakdown();
    stats["load_median_ms"]=quantile(load,.5);
    stats["load_repetitions"]=loads;
    stats["serialized_bytes"]=std::filesystem::file_size(path/"tree.bin");
    stats["paper_model_bytes"]=size.totalBytes();
    stats["paper_model_leaf_bytes"]=size.leafBytes();
    stats["paper_model_internal_bytes"]=size.internalBytes();
    stats["verified_all_queries"]=true;stats["total_result_count"]=result_count;stats["nonempty_queries"]=nonempty;
    report[kind]=stats;
  }
  if(!use_expected) {
    report["bitmap_baseline"]=measure(baseline,queries);
    report["bitmap_baseline"]["paper_model_bytes"]=baseline.estimateIndexSizeBytes();
  }
  std::ofstream out(directory/"query_results.json");out<<report.dump(2)<<"\n";
  std::cout<<report.dump(2)<<"\n";
}
