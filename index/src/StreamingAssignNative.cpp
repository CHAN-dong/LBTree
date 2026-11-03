#include <algorithm>
#include <cstdint>
#include <vector>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif
using I = int64_t;
struct State {
  std::vector<uint8_t> keywords, active;
  std::vector<I> missing;
  I objects=0, omega=0, count=0;
  double sum=0;
  State(I v,I m):keywords(v),active(m),missing(m){}
};
struct Delta { double cost; std::vector<I> keywords, queries; };

// Arrays are canonical CSR int64; output labels preserve sequential decisions.
extern "C" EXPORT int lbtree_stream(
    I n,I v,I m,const I* op,const I* oi,const I* qp,const I* qi,
    const I* hp,const I* hi,const I* sl,I nl,const I* sr,I nr,
    const I* ql,I ml,const I* qr,I mr,
    double epsilon,double alpha,double beta,double gamma,double ws,double wq,
    double balance,int8_t* labels,uint8_t* active_left,uint8_t* active_right) {
  try {
    std::vector<std::vector<I>> inverted(v);
    std::vector<I> lengths(m),qm1(m);
    for(I q=0;q<m;++q) {
      lengths[q]=qp[q+1]-qp[q]; qm1[q]=std::max<I>(0,lengths[q]-1);
      for(I j=qp[q];j<qp[q+1];++j) inverted[qi[j]].push_back(q);
    }
    State left(v,m),right(v,m);
    std::fill(labels,labels+n,int8_t(-1));
    auto initialize=[&](State& s,const I* seeds,I size,const I* qs,I qsize,int side) {
      s.objects=size;
      for(I j=0;j<size;++j) {
        I o=seeds[j]; labels[o]=static_cast<int8_t>(side);
        for(I t=op[o];t<op[o+1];++t) s.keywords[oi[t]]=1;
      }
      for(I k=0;k<v;++k) s.omega+=s.keywords[k];
      for(I q=0;q<m;++q) {
        I missing=0;
        for(I t=qp[q];t<qp[q+1];++t) missing+=!s.keywords[qi[t]];
        s.missing[q]=missing;
      }
      for(I j=0;j<qsize;++j) s.active[qs[j]]=1;
      for(I q=0;q<m;++q) if(s.active[q]) {++s.count;s.sum+=double(qm1[q]);}
    };
    initialize(left,sl,nl,ql,ml,0);initialize(right,sr,nr,qr,mr,1);
    std::vector<I> decrements(m);
    auto evaluate=[&](const State& s,I o) {
      Delta d{};
      for(I j=op[o];j<op[o+1];++j) if(!s.keywords[oi[j]]) d.keywords.push_back(oi[j]);
      double storage=double(s.omega)+double(d.keywords.size())*(double(s.objects)+33.0)+32.0;
      I existing_hits=0;
      for(I j=hp[o];j<hp[o+1];++j) existing_hits+=s.active[hi[j]];
      double existing=s.count ? alpha*s.sum+beta*double(s.count)+gamma*double(existing_hits) : 0.0;
      std::vector<I> touched;
      for(I kw:d.keywords) for(I q:inverted[kw]) if(!s.active[q] && s.missing[q]>0) {
        if(decrements[q]==0) touched.push_back(q);
        ++decrements[q];
      }
      double sum_new=0;
      for(I q:touched) if(s.missing[q]==decrements[q]) {d.queries.push_back(q);sum_new+=double(qm1[q]);}
      I new_hits=0;
      if(!d.queries.empty()) for(I j=hp[o];j<hp[o+1];++j) {
        I q=hi[j];
        if(!s.active[q] && decrements[q]>0 && s.missing[q]==decrements[q]) ++new_hits;
      }
      for(I q:touched) decrements[q]=0;
      double added=epsilon*double(d.queries.size())
          +double(s.objects+1)*(alpha*sum_new+beta*double(d.queries.size()))+gamma*double(new_hits);
      d.cost=ws*storage+wq*(existing+added);
      return d;
    };
    auto update=[&](State& s,const Delta& d) {
      ++s.objects;
      for(I kw:d.keywords) {
        if(s.keywords[kw]) continue;
        s.keywords[kw]=1;++s.omega;
        for(I q:inverted[kw]) if(s.missing[q]>0) --s.missing[q];
      }
      double sum_new=0;
      for(I q:d.queries) {s.active[q]=1;sum_new+=double(qm1[q]);}
      s.count+=I(d.queries.size());s.sum+=sum_new;
    };
    for(I o=0;o<n;++o) if(labels[o]<0) {
      auto dl=evaluate(left,o),dr=evaluate(right,o);
      if(balance>0) {
        I diff=left.objects-right.objects;
        double current=double(diff*diff);
        dl.cost+=balance*(double((diff+1)*(diff+1))-current);
        dr.cost+=balance*(double((diff-1)*(diff-1))-current);
      }
      if(dl.cost<=dr.cost) {labels[o]=0;update(left,dl);} else {labels[o]=1;update(right,dr);}
    }
    std::copy(left.active.begin(),left.active.end(),active_left);
    std::copy(right.active.begin(),right.active.end(),active_right);
    return 0;
  } catch(...) { return -1; }
}
