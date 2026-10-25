#include <array>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <hdf5.h>
#include <torch/torch.h>

#include "eegexpcpp/config.hpp"
#include "eegexpcpp/dataset_h5.hpp"
#include "eegexpcpp/decoder.hpp"
#include "eegexpcpp/encoder.hpp"
#include "eegexpcpp/jvp.hpp"
#include "eegexpcpp/latent_ae_ode.hpp"
#include "eegexpcpp/mlp.hpp"

namespace {

int failures = 0;

void check(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << '\n';
    ++failures;
  }
}

void check_close(const torch::Tensor& a,
                 const torch::Tensor& b,
                 double rtol,
                 double atol,
                 const char* msg) {
  if (!torch::allclose(a, b, rtol, atol)) {
    const auto err = (a - b).abs().max().item<double>();
    std::cerr << "FAIL: " << msg << " (max |diff|=" << err << ")\n";
    ++failures;
  }
}

std::string temp_h5_path() {
  const auto* base = std::getenv("TMPDIR");
  if (base == nullptr) {
    base = "/tmp";
  }
  return (std::filesystem::path(base) / "eegexpcpp_latent_test.h5").string();
}

void write_ft_dataset(hid_t loc,
                      const char* name,
                      const std::vector<float>& data,
                      hsize_t f_dim,
                      hsize_t t_dim) {
  hid_t space = H5Screate_simple(2, std::array<hsize_t, 2>{f_dim, t_dim}.data(),
                                 nullptr);
  hid_t dset = H5Dcreate2(loc, name, H5T_IEEE_F32LE, space, H5P_DEFAULT,
                            H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dset);
  H5Sclose(space);
}

void write_test_h5(const std::string& path, int f_dim, int t_dim) {
  std::vector<float> x(static_cast<size_t>(f_dim * t_dim));
  std::vector<float> dx(static_cast<size_t>(f_dim * t_dim));
  for (int t = 0; t < t_dim; ++t) {
    for (int f = 0; f < f_dim; ++f) {
      const size_t idx = static_cast<size_t>(f) * t_dim + t;
      x[idx] = static_cast<float>(0.01 * f + 0.1 * t);
      dx[idx] = static_cast<float>(0.002 * f - 0.01 * t);
    }
  }
  hid_t file = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    throw std::runtime_error("H5Fcreate failed");
  }
  hid_t g0 = H5Gcreate2(file, "0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  write_ft_dataset(g0, "X", x, f_dim, t_dim);
  write_ft_dataset(g0, "dX", dx, f_dim, t_dim);
  H5Gclose(g0);
  H5Fclose(file);
}

void test_encoder_jvp() {
  torch::manual_seed(0);
  const int64_t b = 2;
  const int64_t t = 7;
  const int64_t f = 4;
  const int64_t l = 3;
  const std::vector<int64_t> enc_hid{8, 8};
  eegexpcpp::Encoder enc(f, l, enc_hid, "gelu");
  auto x = torch::randn({b, t, f}, torch::requires_grad());
  auto dx = torch::randn({b, t, f});
  auto [z, dz] = enc->forward(x, dx);
  check(z.sizes() == torch::IntArrayRef({b, t, l}), "encoder Z shape");
  check(dz.sizes() == torch::IntArrayRef({b, t, l}), "encoder dZ shape");

  auto enc_fn = [&](const at::Tensor& xin) { return enc->encode(xin); };
  auto ref = eegexpcpp::jvp(enc_fn, x, dx).second;
  check_close(dz, ref, 1e-5, 1e-6, "encoder dZ matches JVP");

  const double eps = 1e-3;
  auto numeric =
      (enc->encode(x + eps * dx) - enc->encode(x - eps * dx)) / (2.0 * eps);
  check_close(dz, numeric, 1e-2, 1e-3, "encoder dZ vs central difference");
}

void test_latent_forward_loss_grad() {
  torch::manual_seed(1);
  const int64_t b = 1;
  const int64_t t = 9;
  const int64_t f = 5;
  const int64_t l = 3;
  eegexpcpp::ModelConfig mcfg;
  mcfg.n_latent = l;
  mcfg.feature_dim = f;
  mcfg.encoder_hidden = std::vector<int64_t>{16};
  mcfg.decoder_hidden = std::vector<int64_t>{16};
  mcfg.ode_hidden = std::vector<int64_t>{8};
  eegexpcpp::LatentAeOde model(mcfg);
  auto x = torch::randn({b, t, f}, torch::requires_grad());
  auto dx = torch::randn({b, t, f});
  auto out = model->forward(x, dx);
  check(out.y.sizes() == x.sizes(), "Y shape");
  check(out.dy_pred.sizes() == dx.sizes(), "dY_pred shape");
  check(out.z.sizes() == torch::IntArrayRef({b, t, l}), "Z shape");

  eegexpcpp::LossConfig lcfg;
  auto losses = model->compute_losses(out, x, dx, lcfg);
  auto total = model->total_loss(losses, lcfg);
  check(std::isfinite(total.item<double>()), "total loss finite");
  total.backward();
  bool any_grad = false;
  for (const auto& p : model->parameters()) {
    if (p.grad().defined() && p.grad().abs().sum().item<double>() > 0.0) {
      any_grad = true;
      break;
    }
  }
  check(any_grad, "encoder/decoder/ode receive gradients");

  auto z0 = out.z.select(1, 0);
  auto x0 = x.select(1, 0).narrow(-1, 0, l);
  check_close(losses.at("J_z_1"), torch::mse_loss(z0, x0), 1e-6, 1e-7,
              "J_z_1 equals mse at t=0");
}

void test_timemlp_jvp_activations() {
  torch::manual_seed(3);
  const int64_t b = 2;
  const int64_t t = 5;
  const int64_t in_f = 4;
  const int64_t out_f = 3;
  const std::vector<int64_t> hid{6};
  const char* acts[] = {"gelu", "relu", "tanh"};
  for (const char* act : acts) {
    eegexpcpp::TimeMlp mlp(in_f, hid, out_f, act);
    auto x = torch::randn({b, t, in_f});
    auto dx = torch::randn({b, t, in_f});
    auto [y, dy] = mlp->forward_jvp(x, dx);
    check(y.sizes() == torch::IntArrayRef({b, t, out_f}), act);
    check_close(y, mlp->forward(x), 1e-6, 1e-7, "TimeMlp primal");
    auto mlp_fn = [&](const at::Tensor& xin) { return mlp->forward(xin); };
    auto ref = eegexpcpp::jvp(mlp_fn, x, dx).second;
    check_close(dy, ref, 1e-5, 1e-6, act);
  }
}

void test_decoder_jvp_chain() {
  torch::manual_seed(2);
  const int64_t b = 1;
  const int64_t t = 6;
  const int64_t f = 5;
  const int64_t l = 3;
  const std::vector<int64_t> dec_hid{12};
  eegexpcpp::Decoder dec(l, f, dec_hid, "gelu");
  auto z = torch::randn({b, t, l}, torch::requires_grad());
  auto dz_pred = torch::randn({b, t, l});
  auto [y, dy] = dec->forward(z, dz_pred);
  check(y.sizes() == torch::IntArrayRef({b, t, f}), "decoder Y shape");
  check(dy.sizes() == torch::IntArrayRef({b, t, f}), "decoder dY shape");
  auto dec_fn = [&](const at::Tensor& zin) { return dec->decode(zin); };
  auto ref = eegexpcpp::jvp(dec_fn, z, dz_pred).second;
  check_close(dy, ref, 1e-5, 1e-6, "decoder dY_pred matches JVP");
}

void test_h5_loader() {
  const int f_dim = 5;
  const int t_dim = 11;
  const auto path = temp_h5_path();
  write_test_h5(path, f_dim, t_dim);
  auto [x, dx] = eegexpcpp::load_h5_pair_btf(path, "0");
  check(x.dim() == 3 && x.size(0) == 1, "h5 batch dim");
  check(x.size(1) == t_dim && x.size(2) == f_dim, "h5 B_T_F layout");
  const float expect = 0.01f * 0 + 0.1f * 3;
  check(std::abs(x.index({0, 3, 0}).item<float>() - expect) < 1e-5f,
        "h5 X sample value");
  std::filesystem::remove(path);
}

void test_encoder_jvp_cuda_dx_on_cpu() {
  if (!torch::cuda::is_available()) {
    return;
  }
  torch::manual_seed(4);
  const int64_t b = 1;
  const int64_t t = 4;
  const int64_t f = 3;
  const int64_t l = 2;
  eegexpcpp::Encoder enc(f, l, std::vector<int64_t>{5}, "gelu");
  enc->to(torch::kCUDA);
  auto x = torch::randn({b, t, f}, torch::TensorOptions().device(torch::kCUDA));
  auto dx = torch::randn({b, t, f});
  auto [z, dz] = enc->forward(x, dx);
  check(z.device().is_cuda(), "encoder Z on cuda");
  check(dz.device().is_cuda(), "encoder dZ on cuda");
}

void test_config_json() {
  const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                    "configs" / "latent_ae_ode.lorenz.json";
  const auto cfg = eegexpcpp::load_config(path.string());
  check(cfg.train.lr > 0.0, "config lr");
  check(!cfg.data.h5_path.empty(), "config h5_path");
  check(cfg.model.n_latent == 3, "config lorenz n_latent");
}

}  // namespace

int main() {
  try {
    test_encoder_jvp();
    test_timemlp_jvp_activations();
    test_encoder_jvp_cuda_dx_on_cpu();
    test_decoder_jvp_chain();
    test_latent_forward_loss_grad();
    test_h5_loader();
    test_config_json();
  } catch (const std::exception& ex) {
    std::cerr << "FAIL: exception: " << ex.what() << '\n';
    return 1;
  }
  if (failures > 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "latent_ae_ode_test: ok\n";
  return 0;
}
