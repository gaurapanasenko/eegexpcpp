#include "eegexpcpp/mlp.hpp"

#include <ATen/Functions.h>

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

std::pair<torch::Tensor, torch::Tensor> linear_jvp(
    const torch::Tensor& x,
    const torch::Tensor& dx,
    torch::nn::LinearImpl& linear) {
  const auto y = linear.forward(x);
  const auto& w = linear.weight;
  const auto dy = dx.matmul(w.transpose(0, 1));
  return {y, dy};
}

std::pair<torch::Tensor, torch::Tensor> activation_jvp(
    const torch::Tensor& pre,
    const torch::Tensor& d_pre,
    const torch::nn::Module& module) {
  if (module.as<torch::nn::GELU>() != nullptr) {
    return {torch::gelu(pre), at::gelu_backward(d_pre, pre)};
  }
  if (module.as<torch::nn::ReLU>() != nullptr) {
    return {torch::relu(pre), d_pre * (pre > 0)};
  }
  if (module.as<torch::nn::Tanh>() != nullptr) {
    const auto y = torch::tanh(pre);
    return {y, d_pre * (1 - y * y)};
  }
  throw std::runtime_error("forward_jvp: unknown activation module");
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

std::pair<torch::Tensor, torch::Tensor> TimeMlpImpl::forward_jvp(
    torch::Tensor x,
    torch::Tensor dx) {
  const auto b = x.size(0);
  const auto t = x.size(1);
  const auto in_f = x.size(2);
  x = x.reshape({b * t, in_f});
  dx = dx.reshape({b * t, in_f}).to(x.device(), x.scalar_type());

  for (const auto& child : seq_->children()) {
    if (auto* linear = child->as<torch::nn::Linear>()) {
      std::tie(x, dx) = linear_jvp(x, dx, *linear);
      continue;
    }
    std::tie(x, dx) = activation_jvp(x, dx, *child);
  }
  const auto out_f = x.size(1);
  return {x.reshape({b, t, out_f}), dx.reshape({b, t, out_f})};
}

}  // namespace eegexpcpp
