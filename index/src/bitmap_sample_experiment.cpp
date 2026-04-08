#include "BitmapIndexBaseline.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include <cmath>
// -----------------------------
// Median utility
// -----------------------------
static int64_t median_ns(std::vector<int64_t>& xs) {
  if (xs.empty()) return 0;
  std::sort(xs.begin(), xs.end());
  size_t n = xs.size();
  if (n & 1) return xs[n / 2];
  return (xs[n / 2 - 1] + xs[n / 2]) / 2;
}

static std::vector<int> make_linear_grid_int(int start, int end, int n_points) {
  std::vector<int> xs;
  if (n_points <= 0) return xs;
  if (n_points == 1) {
    xs.push_back(start);
    return xs;
  }
  xs.reserve((size_t)n_points);
  for (int i = 0; i < n_points; ++i) {
    double t = (double)i / (double)(n_points - 1);
    int v = (int)std::llround((1.0 - t) * (double)start + t * (double)end);
    if (!xs.empty() && v == xs.back()) v = xs.back() + 1;
    if (v < start) v = start;
    if (v > end) v = end;
    xs.push_back(v);
  }
  // ensure monotonic non-decreasing
  for (size_t i = 1; i < xs.size(); ++i) {
    if (xs[i] < xs[i - 1]) xs[i] = xs[i - 1];
  }
  return xs;
}

// -----------------------------
// Synthetic dataset spec
// -----------------------------
struct SyntheticDatasetSpec {
  int num_universal_terms = 8;      // must be >= 7 for qk up to 8
  int max_target_result = 200;     // anchor covers 0..max_target_result
  int filler_terms_per_object = 4;  // background terms per object
};

// Unified spec (IMPORTANT: train & plot testset share the same mechanism)
static SyntheticDatasetSpec getUnifiedSyntheticSpec() {
  SyntheticDatasetSpec spec;
  spec.num_universal_terms = 8;
  spec.max_target_result = 200;
  spec.filler_terms_per_object = 4;
  return spec;
}

// -----------------------------
// Generate controlled objects in memory
//
// Term id layout:
// [0..U-1]                 : universal terms (in all objects)
// [U..U+RMAX]              : anchor terms for rc=0..RMAX (anchor_0 absent)
// [U+RMAX+1..]             : filler terms (deterministic)
// anchor_r appears in first r objects => query containing anchor_r returns ~r results
// -----------------------------
static std::vector<std::vector<int>> generateSyntheticObjectsControlled(
    int num_objects,
    const SyntheticDatasetSpec& spec
) {
  if (num_objects < 1) throw std::runtime_error("num_objects must be >= 1");
  if (spec.num_universal_terms < 7) throw std::runtime_error("num_universal_terms must be >= 7");

  const int U = spec.num_universal_terms;
  const int RMAX = spec.max_target_result;
  const int F = spec.filler_terms_per_object;

  const int anchor_base = U;
  const int filler_base = U + (RMAX + 1);

  std::vector<std::vector<int>> objects((size_t)num_objects);

  for (int oid = 0; oid < num_objects; ++oid) {
    auto& obj = objects[(size_t)oid];
    obj.reserve((size_t)U + (size_t)F + 16);

    // universal
    for (int t = 0; t < U; ++t) obj.push_back(t);

    // anchors: add anchor_r if oid < r, for r=1..min(RMAX,num_objects)
    const int max_r_here = std::min(RMAX, num_objects);
    for (int r = 1; r <= max_r_here; ++r) {
      if (oid < r) obj.push_back(anchor_base + r);
    }
    // anchor_0 (anchor_base+0) is absent from all objects => rc ~ 0

    // filler terms
    for (int k = 0; k < F; ++k) {
      int filler_id = filler_base + oid * F + k;
      obj.push_back(filler_id);
    }

    std::sort(obj.begin(), obj.end());
    obj.erase(std::unique(obj.begin(), obj.end()), obj.end());
  }

  return objects;
}

// -----------------------------
// Controlled query: [anchor(rc)] + (qk-1) universal terms
// This keeps result size controlled by rc while changing qk.
// -----------------------------
static std::vector<int> makeControlledQuery(
    int query_keyword_count,
    int target_result_count,
    const SyntheticDatasetSpec& spec
) {
  if (query_keyword_count < 2 || query_keyword_count > 8) {
    throw std::runtime_error("query_keyword_count must be in [2,8]");
  }
  if (target_result_count < 0 || target_result_count > spec.max_target_result) {
    throw std::runtime_error("target_result_count out of range");
  }
  const int U = spec.num_universal_terms;
  const int anchor_base = U;

  std::vector<int> q;
  q.reserve((size_t)query_keyword_count);

  q.push_back(anchor_base + target_result_count);
  for (int i = 0; i < query_keyword_count - 1; ++i) q.push_back(i);

  return q; // keep deterministic order
}

// -----------------------------
// Append plot testset (with plot_type col) into training CSV (without plot_type)
// plot csv format:
// plot_type,num_objects,dataset_total_keyword_occurrences,dataset_distinct_keywords,query_keyword_count,result_count,query_time_ns
// train csv format:
// num_objects,dataset_total_keyword_occurrences,dataset_distinct_keywords,query_keyword_count,result_count,query_time_ns
// -----------------------------
static void appendPlotTestsetToTrainingCSV(
    const std::string& train_csv,
    const std::string& plot_csv
) {
  std::ifstream in_plot(plot_csv);
  if (!in_plot) throw std::runtime_error("Cannot open plot csv: " + plot_csv);

  std::ofstream out_train(train_csv, std::ios::app);
  if (!out_train) throw std::runtime_error("Cannot open train csv for append: " + train_csv);

  std::string line;
  bool first_line = true;
  size_t appended = 0;

  while (std::getline(in_plot, line)) {
    if (line.empty()) continue;
    if (first_line) { first_line = false; continue; } // skip header

    auto p = line.find(',');
    if (p == std::string::npos) continue;
    std::string tail = line.substr(p + 1); // remove plot_type,
    out_train << tail << "\n";
    ++appended;
  }

  out_train.flush();
  std::cout << "Appended " << appended << " plot rows into training CSV: " << train_csv << std::endl;
}

// -----------------------------
// Generate base training samples (grid over N, qk, rc)
// Writes: bitmap_samples_controlled.csv
// Columns:
// num_objects,dataset_total_keyword_occurrences,dataset_distinct_keywords,query_keyword_count,result_count,query_time_ns
// -----------------------------
static void runBitmapSampleExperimentInMemory(const std::string& out_csv) {
  // NOTE: running N=1..100000 (every integer) is too slow.
  // use a reasonable grid; you can refine later.
  const std::vector<int> num_objects_grid = {
      10, 20, 50, 100, 200, 500, 1000,
      2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000
  };

  const int qk_min = 2;
  const int qk_max = 8;

  // IMPORTANT: training rc range can be wide; keep 0..200 to match your current requirement
  const int rc_min = 0;
  const int rc_max = 200;

  const int warmup_per_dataset = 200;
  const int repeat_per_query = 3;

  SyntheticDatasetSpec spec = getUnifiedSyntheticSpec();

  std::ofstream out(out_csv);
  if (!out) throw std::runtime_error("Cannot open output csv: " + out_csv);

  out << "num_objects,"
      << "dataset_total_keyword_occurrences,"
      << "dataset_distinct_keywords,"
      << "query_keyword_count,"
      << "result_count,"
      << "query_time_ns\n";

  for (int N : num_objects_grid) {
    std::cout << "[Train samples] Build dataset N=" << N << std::endl;

    auto objects = generateSyntheticObjectsControlled(N, spec);

    BitmapIndexBaseline bix;
    bix.buildFromObjects(objects);

    // warmup
    {
      auto warm_q = makeControlledQuery(4, 10, spec);
      for (int i = 0; i < warmup_per_dataset; ++i) (void)bix.query(warm_q);
    }

    for (int qk = qk_min; qk <= qk_max; ++qk) {
      for (int rc = rc_min; rc <= rc_max; ++rc) {
        auto q = makeControlledQuery(qk, rc, spec);

        std::vector<int64_t> times_ns;
        times_ns.reserve((size_t)repeat_per_query);
        std::vector<int64_t> ans;

        for (int rep = 0; rep < repeat_per_query; ++rep) {
          auto t0 = std::chrono::steady_clock::now();
          ans = bix.query(q);
          auto t1 = std::chrono::steady_clock::now();
          times_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        }

        int64_t med = median_ns(times_ns);

        out << bix.numObjects() << ","
            << bix.totalObjectKeywordOccurrences() << ","
            << bix.distinctKeywords() << ","
            << qk << ","
            << (int64_t)ans.size() << ","
            << med << "\n";
      }
    }
  }

  out.close();
  std::cout << "Saved training samples CSV: " << out_csv << std::endl;
}

// -----------------------------
// Generate controlled plot testset CSV (3 plots * 100 points)
//
// Output columns:
// plot_type,num_objects,dataset_total_keyword_occurrences,dataset_distinct_keywords,query_keyword_count,result_count,query_time_ns
//
// Control defaults (as you requested):
// default N = 100000
// default qk = 4
// default rc = 10
//
// Variable ranges:
// - result_count plot: 0..200 (as you requested)
// - num_objects plot: 10..100000 (100 points)
// - query_keyword_count plot: values 2..8 repeated to reach 100 points
// -----------------------------
static void generateControlledPlotTestsetCSV(
    const std::string& out_csv,
    int repeat_per_point = 3
) {
  const int DEFAULT_N  = 100000;
  const int DEFAULT_QK = 4;
  const int DEFAULT_RC = 10;

  const int POINTS_PER_PLOT = 100;
  const int RC_MIN = 0;
  const int RC_MAX = 200;

  SyntheticDatasetSpec spec = getUnifiedSyntheticSpec();

  std::ofstream out(out_csv);
  if (!out) throw std::runtime_error("Cannot open output csv: " + out_csv);

  out << "plot_type,"
      << "num_objects,"
      << "dataset_total_keyword_occurrences,"
      << "dataset_distinct_keywords,"
      << "query_keyword_count,"
      << "result_count,"
      << "query_time_ns\n";

  size_t written_rows = 0;

  // ---- Plot 1: x=num_objects, keep qk=4, rc=10 ----
  {
    auto N_grid = make_linear_grid_int(/*start=*/10, /*end=*/100000, POINTS_PER_PLOT);
    for (int N : N_grid) {
      auto objects = generateSyntheticObjectsControlled(N, spec);
      BitmapIndexBaseline bix;
      bix.buildFromObjects(objects);

      auto q = makeControlledQuery(DEFAULT_QK, DEFAULT_RC, spec);

      std::vector<int64_t> tns;
      tns.reserve((size_t)repeat_per_point);
      std::vector<int64_t> ans;

      for (int r = 0; r < repeat_per_point; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ans = bix.query(q);
        auto t1 = std::chrono::steady_clock::now();
        tns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      }
      int64_t med = median_ns(tns);

      out << "num_objects" << ","
          << bix.numObjects() << ","
          << bix.totalObjectKeywordOccurrences() << ","
          << bix.distinctKeywords() << ","
          << DEFAULT_QK << ","
          << (int64_t)ans.size() << ","
          << med << "\n";
      ++written_rows;
    }
    std::cout << "[Plot testset] Finished plot_type=num_objects (100 points)\n";
  }

  // ---- Plot 2: x=query_keyword_count, keep N=100000, rc=10 ----
  {
    auto objects = generateSyntheticObjectsControlled(DEFAULT_N, spec);
    BitmapIndexBaseline bix;
    bix.buildFromObjects(objects);

    for (int i = 0; i < POINTS_PER_PLOT; ++i) {
      int qk = 2 + (i % 7); // 2..8 repeat
      auto q = makeControlledQuery(qk, DEFAULT_RC, spec);

      std::vector<int64_t> tns;
      tns.reserve((size_t)repeat_per_point);
      std::vector<int64_t> ans;

      for (int r = 0; r < repeat_per_point; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ans = bix.query(q);
        auto t1 = std::chrono::steady_clock::now();
        tns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      }
      int64_t med = median_ns(tns);

      out << "query_keyword_count" << ","
          << bix.numObjects() << ","
          << bix.totalObjectKeywordOccurrences() << ","
          << bix.distinctKeywords() << ","
          << qk << ","
          << (int64_t)ans.size() << ","
          << med << "\n";
      ++written_rows;
    }
    std::cout << "[Plot testset] Finished plot_type=query_keyword_count (100 points)\n";
  }

  // ---- Plot 3: x=result_count, keep N=100000, qk=4, rc in [0,200] ----
  {
    auto objects = generateSyntheticObjectsControlled(DEFAULT_N, spec);
    BitmapIndexBaseline bix;
    bix.buildFromObjects(objects);

    auto rc_grid = make_linear_grid_int(RC_MIN, RC_MAX, POINTS_PER_PLOT);

    for (int rc : rc_grid) {
      auto q = makeControlledQuery(DEFAULT_QK, rc, spec);

      std::vector<int64_t> tns;
      tns.reserve((size_t)repeat_per_point);
      std::vector<int64_t> ans;

      for (int r = 0; r < repeat_per_point; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        ans = bix.query(q);
        auto t1 = std::chrono::steady_clock::now();
        tns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      }
      int64_t med = median_ns(tns);

      out << "result_count" << ","
          << bix.numObjects() << ","
          << bix.totalObjectKeywordOccurrences() << ","
          << bix.distinctKeywords() << ","
          << DEFAULT_QK << ","
          << (int64_t)ans.size() << ","
          << med << "\n";
      ++written_rows;
    }
    std::cout << "[Plot testset] Finished plot_type=result_count (100 points)\n";
  }

  out.close();
  std::cout << "Controlled plot testset rows written = " << written_rows << std::endl;
  std::cout << "Saved controlled plot testset CSV: " << out_csv << std::endl;
}

// -----------------------------
// Append dense num_objects slice training samples (fix qk=4, rc=10)
// This is to help the num_objects slice fitting.
// -----------------------------
static void appendDenseNumObjectsSliceTrainingSamples(
    const std::string& train_csv,
    int repeat_per_point = 3
) {
  const int FIXED_QK = 4;
  const int FIXED_RC = 10;

  SyntheticDatasetSpec spec = getUnifiedSyntheticSpec();

  std::vector<int> N_grid;
  N_grid.reserve(3000);

  for (int N = 10; N <= 2000; N += 10) N_grid.push_back(N);
  for (int N = 2050; N <= 10000; N += 50) N_grid.push_back(N);
  for (int N = 10500; N <= 50000; N += 500) N_grid.push_back(N);
  for (int N = 51000; N <= 100000; N += 1000) N_grid.push_back(N);

  std::sort(N_grid.begin(), N_grid.end());
  N_grid.erase(std::unique(N_grid.begin(), N_grid.end()), N_grid.end());

  std::ofstream out(train_csv, std::ios::app);
  if (!out) throw std::runtime_error("Cannot open training csv for append: " + train_csv);

  std::cout << "[Dense N slice] Appending " << N_grid.size()
            << " points into " << train_csv << std::endl;

  for (size_t i = 0; i < N_grid.size(); ++i) {
    int N = N_grid[i];

    auto objects = generateSyntheticObjectsControlled(N, spec);
    BitmapIndexBaseline bix;
    bix.buildFromObjects(objects);

    auto q = makeControlledQuery(FIXED_QK, FIXED_RC, spec);

    std::vector<int64_t> times_ns;
    times_ns.reserve((size_t)repeat_per_point);
    std::vector<int64_t> ans;

    for (int r = 0; r < repeat_per_point; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      ans = bix.query(q);
      auto t1 = std::chrono::steady_clock::now();
      times_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    int64_t med_ns = median_ns(times_ns);

    out << bix.numObjects() << ","
        << bix.totalObjectKeywordOccurrences() << ","
        << bix.distinctKeywords() << ","
        << FIXED_QK << ","
        << (int64_t)ans.size() << ","
        << med_ns << "\n";

    if ((i + 1) % 100 == 0) {
      std::cout << "  progress: " << (i + 1) << "/" << N_grid.size() << std::endl;
    }
  }

  out.flush();
  std::cout << "[Dense N slice] Done.\n";
}

// -----------------------------
// main
// -----------------------------
int main() {
  try {
    const std::string train_csv = "/root/dong_11.02/MLKQI1.29/dataset/datasets_for_costmodel/bitmap_samples_controlled.csv";
    const std::string plot_csv  = "/root/dong_11.02/MLKQI1.29/dataset/datasets_for_costmodel/bitmap_plot_testset_controlled.csv";

    // 1) Generate base training samples
    runBitmapSampleExperimentInMemory(train_csv);

    // 2) Generate controlled plot testset
    generateControlledPlotTestsetCSV(plot_csv, /*repeat_per_point=*/3);

    // 3) Append plot testset into training set (optional but you asked)
    appendPlotTestsetToTrainingCSV(train_csv, plot_csv);

    // 4) Append dense num_objects slice samples
    appendDenseNumObjectsSliceTrainingSamples(train_csv, /*repeat_per_point=*/3);

    std::cout << "All done.\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}