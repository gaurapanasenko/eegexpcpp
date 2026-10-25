#include "eegexpcpp/mlp.hpp"

#include <stdexcept>

namespace eegexpcpp {

namespace {

void push_activation(torch::nn::Sequential& seq, const std::string& name) {
  if (name == "gelu") {
    seq->push_back(torch::nn::GELU());
    return;
  }
  if (name == "relu") {
    seq->push_back(torch::nn::ReLU());
    return;
  }
  if (name == "tanh") {
    seq->push_back(torch::nn::Tanh());
    return;
  }
  throw std::runtime_error("unknown activation: " + name);
}

}  // namespace

TimeMlpImpl::TimeMlpImpl(int64_t in_features,
                         const std::vector<int64_t>& hidden,
                         int64_t out_features,
                         const std::string& activation)
    : seq_(torch::nn::Sequential()) {
  int64_t cur = in_features;
  for (int64_t h : hidden) {
    seq_->push_back(torch::nn::Linear(
        torch::nn::LinearOptions(cur, h).bias(true)));
    push_activation(seq_, activation);
    cur = h;
  }
  seq_->push_back(torch::nn::Linear(
      torch::nn::LinearOptions(cur, out_features).bias(true)));
  register_module("seq", seq_);
}

torch::Tensor TimeMlpImpl::forward(torch::Tensor x) {
  const auto b = x.size(0);
  const auto t = x.size(1);
  const auto in_f = x.size(2);
  x = x.reshape({b * t, in_f});
  x = seq_->forward(x);
  return x.reshape({b, t, -1});
}

}  // namespace eegexpcpp
