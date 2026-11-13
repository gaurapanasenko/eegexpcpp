#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <doctest.h>
#include <hdf5.h>
#include <torch/torch.h>

#include "eegexpcpp/config.hpp"
#include "eegexpcpp/dataset_h5.hpp"
#include "eegexpcpp/decoder.hpp"
#include "eegexpcpp/encoder.hpp"
#include "eegexpcpp/jvp.hpp"
#include "eegexpcpp/latent_ae_ode.hpp"
#include "eegexpcpp/mlp.hpp"
#include "eegexpcpp/ode.hpp"
#include "eegexpcpp/ode_sindy.hpp"
#include "eegexpcpp/rk4.hpp"
#include "eegexpcpp/sindy_library.hpp"
#include "eegexpcpp/sindy_xi.hpp"
#include "eegexpcpp/sindy_table.hpp"

namespace {

void check_close(const torch::Tensor& a,
                 const torch::Tensor& b,
                 double rtol,
                 double atol,
                 const char* msg) {
  DOCTEST_MESSAGE("context: ", msg);
  DOCTEST_CHECK(torch::allclose(a, b, rtol, atol));
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

void write_scalar_dataset(hid_t loc, const char* name, float value) {
  hid_t space = H5Screate(H5S_SCALAR);
  hid_t dset = H5Dcreate2(loc, name, H5T_IEEE_F32LE, space, H5P_DEFAULT,
                          H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
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
  write_scalar_dataset(g0, "X_mean", 1.5f);
  write_scalar_dataset(g0, "X_std", 2.0f);
  H5Gclose(g0);
  H5Fclose(file);
}

}  // namespace

DOCTEST_TEST_CASE("encoder jvp") {
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
  DOCTEST_CHECK(z.sizes() == torch::IntArrayRef({b, t, l}));
  DOCTEST_CHECK(dz.sizes() == torch::IntArrayRef({b, t, l}));

  auto enc_fn = [&](const at::Tensor& xin) { return enc->encode(xin); };
  auto ref = eegexpcpp::jvp(enc_fn, x, dx).second;
  check_close(dz, ref, 1e-5, 1e-6, "encoder dZ matches JVP");

  const double eps = 1e-3;
  auto numeric =
      (enc->encode(x + eps * dx) - enc->encode(x - eps * dx)) / (2.0 * eps);
  check_close(dz, numeric, 1e-2, 1e-3, "encoder dZ vs central difference");
}

DOCTEST_TEST_CASE("latent forward loss grad") {
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
  mcfg.ode.nn.hidden = std::vector<int64_t>{8};
  eegexpcpp::LatentAeOde model(mcfg);
  auto x = torch::randn({b, t, f}, torch::requires_grad());
  auto dx = torch::randn({b, t, f});
  auto out = model->forward(x, dx);
  DOCTEST_CHECK(out.y.sizes() == x.sizes());
  DOCTEST_CHECK(out.dy_pred.sizes() == dx.sizes());
  DOCTEST_CHECK(out.z.sizes() == torch::IntArrayRef({b, t, l}));

  eegexpcpp::LossConfig lcfg;
  auto losses = model->compute_losses(out, x, dx, lcfg);
  auto total = model->total_loss(losses, lcfg);
  DOCTEST_CHECK(std::isfinite(total.item<double>()));
  total.backward();
  bool any_grad = false;
  for (const auto& p : model->parameters()) {
    if (p.grad().defined() && p.grad().abs().sum().item<double>() > 0.0) {
      any_grad = true;
      break;
    }
  }
  DOCTEST_CHECK(any_grad);

  check_close(
      losses.at("J_z_1"),
      torch::mse_loss(out.z.select(2, 0), x.select(2, 0)),
      1e-6, 1e-7, "J_z_1 equals mse(z[:,:,0], x[:,:,0])");
}

DOCTEST_TEST_CASE("timemlp jvp activations") {
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
    DOCTEST_CHECK(y.sizes() == torch::IntArrayRef({b, t, out_f}));
    check_close(y, mlp->forward(x), 1e-6, 1e-7, "TimeMlp primal");
    auto mlp_fn = [&](const at::Tensor& xin) { return mlp->forward(xin); };
    auto ref = eegexpcpp::jvp(mlp_fn, x, dx).second;
    check_close(dy, ref, 1e-5, 1e-6, act);
  }
}

DOCTEST_TEST_CASE("decoder jvp chain") {
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
  DOCTEST_CHECK(y.sizes() == torch::IntArrayRef({b, t, f}));
  DOCTEST_CHECK(dy.sizes() == torch::IntArrayRef({b, t, f}));
  auto dec_fn = [&](const at::Tensor& zin) { return dec->decode(zin); };
  auto ref = eegexpcpp::jvp(dec_fn, z, dz_pred).second;
  check_close(dy, ref, 1e-5, 1e-6, "decoder dY_pred matches JVP");
}

DOCTEST_TEST_CASE("h5 loader") {
  const int f_dim = 5;
  const int t_dim = 11;
  const auto path = temp_h5_path();
  write_test_h5(path, f_dim, t_dim);
  auto [x, dx] = eegexpcpp::load_h5_pair_btf(path, "0");
  DOCTEST_CHECK(x.dim() == 3);
  DOCTEST_CHECK(x.size(0) == 1);
  DOCTEST_CHECK(x.size(1) == t_dim);
  DOCTEST_CHECK(x.size(2) == f_dim);
  const float expect = 0.01f * 0 + 0.1f * 3;
  DOCTEST_CHECK(std::abs(x.index({0, 3, 0}).item<float>() - expect) < 1e-5f);
  const auto [mean, std] = eegexpcpp::load_h5_z_affine(path, "0");
  DOCTEST_CHECK(mean == doctest::Approx(1.5f));
  DOCTEST_CHECK(std == doctest::Approx(2.0f));
  std::filesystem::remove(path);
}

DOCTEST_TEST_CASE("latent z affine broadcast") {
  eegexpcpp::ModelConfig mcfg;
  mcfg.n_latent = 3;
  mcfg.feature_dim = 5;
  eegexpcpp::LatentAeOde model(mcfg);
  model->set_latent_z_affine(1.0f, 2.0f);
  const float scale = 2.0f + 1e-8f;
  auto z = torch::full({1, 4, 3}, 0.5f);
  check_close(model->latent_z_physical(z),
              torch::full_like(z, 0.5f * scale + 1.0f), 1e-6, 1e-7,
              "z_phys = z*scale+mean");
  auto dz = torch::ones_like(z);
  check_close(model->latent_dz_physical(dz), torch::full_like(z, scale), 1e-6,
              1e-7, "dz_phys = dz*scale");
}

DOCTEST_TEST_CASE("encoder jvp cuda dx on cpu") {
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
  DOCTEST_CHECK(z.device().is_cuda());
  DOCTEST_CHECK(dz.device().is_cuda());
}

DOCTEST_TEST_CASE("sindy feature library names") {
  eegexpcpp::SindyLibraryConfig poly;
  poly.kind = eegexpcpp::SindyLibraryConfig::Kind::Poly;
  poly.poly.degree = 1;
  eegexpcpp::SindyFeatureLibrary lib(2, {poly});
  DOCTEST_CHECK(lib.size() == 3);
  const auto& names = lib.feature_names();
  DOCTEST_CHECK(names[0] == "1");
  DOCTEST_CHECK(names[1] == "x0");
  DOCTEST_CHECK(names[2] == "x1");
}

DOCTEST_TEST_CASE("sindy poly feature count") {
  eegexpcpp::PolyLibraryConfig poly;
  DOCTEST_CHECK(eegexpcpp::poly_feature_count(2, poly) == 6);
  poly.include_interaction = false;
  DOCTEST_CHECK(eegexpcpp::poly_feature_count(2, poly) == 5);
  poly.interaction_only = true;
  poly.include_interaction = true;
  poly.degree = 2;
  DOCTEST_CHECK(eegexpcpp::poly_feature_count(3, poly) == 7);
}

DOCTEST_TEST_CASE("sindy ode forward") {
  torch::manual_seed(5);
  const int64_t b = 1;
  const int64_t t = 4;
  const int64_t l = 2;
  eegexpcpp::OdeConfig ocfg;
  ocfg.type = eegexpcpp::OdeType::Sindy;
  eegexpcpp::SindyLibraryConfig poly;
  poly.kind = eegexpcpp::SindyLibraryConfig::Kind::Poly;
  poly.poly.degree = 1;
  ocfg.libraries.push_back(poly);
  eegexpcpp::Ode ode(l, ocfg);
  auto z = torch::randn({b, t, l});
  auto dz = ode->forward(z);
  DOCTEST_CHECK(dz.sizes() == torch::IntArrayRef({b, t, l}));
  const int64_t f = eegexpcpp::poly_feature_count(l, poly.poly);
  DOCTEST_CHECK(f == 3);
}

DOCTEST_TEST_CASE("latent forward sindy") {
  torch::manual_seed(6);
  const int64_t b = 1;
  const int64_t t = 5;
  const int64_t f = 4;
  const int64_t l = 2;
  eegexpcpp::ModelConfig mcfg;
  mcfg.n_latent = l;
  mcfg.feature_dim = f;
  mcfg.encoder_hidden = std::vector<int64_t>{8};
  mcfg.decoder_hidden = std::vector<int64_t>{8};
  mcfg.ode.type = eegexpcpp::OdeType::Sindy;
  eegexpcpp::SindyLibraryConfig flib;
  flib.kind = eegexpcpp::SindyLibraryConfig::Kind::Fourier;
  mcfg.ode.libraries.push_back(flib);
  eegexpcpp::LatentAeOde model(mcfg);
  auto x = torch::randn({b, t, f});
  auto dx = torch::randn({b, t, f});
  auto out = model->forward(x, dx);
  DOCTEST_CHECK(out.dz_pred.sizes() == torch::IntArrayRef({b, t, l}));
  eegexpcpp::LossConfig lcfg;
  auto total = model->total_loss(model->compute_losses(out, x, dx, lcfg), lcfg);
  DOCTEST_CHECK(std::isfinite(total.item<double>()));
}

DOCTEST_TEST_CASE("sindy xi starts at zero") {
  eegexpcpp::OdeConfig ocfg;
  ocfg.type = eegexpcpp::OdeType::Sindy;
  eegexpcpp::SindyLibraryConfig poly;
  poly.kind = eegexpcpp::SindyLibraryConfig::Kind::Poly;
  poly.poly.degree = 1;
  ocfg.libraries.push_back(poly);
  eegexpcpp::OdeSindy sindy(2, ocfg.libraries);
  DOCTEST_CHECK(torch::allclose(sindy->xi_matrix(), torch::zeros({3, 2})));
}

DOCTEST_TEST_CASE("sindy xi apply threshold") {
  auto xi = torch::tensor({{0.2, 0.01}, {-0.15, 0.3}});
  const auto out = eegexpcpp::xi_apply_threshold(xi, 0.1);
  DOCTEST_CHECK(out[0][0].item<double>() == doctest::Approx(0.2));
  DOCTEST_CHECK(out[0][1].item<double>() == doctest::Approx(0.0));
  DOCTEST_CHECK(out[1][0].item<double>() == doctest::Approx(-0.15));
  DOCTEST_CHECK(eegexpcpp::xi_count_nnz(out, 0.1) == 3);
}

DOCTEST_TEST_CASE("sindy xi frozen trainable mask") {
  auto trainable = torch::ones({2, 2});
  const auto pruned =
      eegexpcpp::xi_apply_threshold(torch::tensor({{0.2, 0.01}, {0.0, 0.3}}),
                                    0.1);
  eegexpcpp::xi_shrink_trainable_mask(trainable, pruned, 0.1);
  DOCTEST_CHECK(trainable[0][0].item<double>() == doctest::Approx(1.0));
  DOCTEST_CHECK(trainable[0][1].item<double>() == doctest::Approx(0.0));
  auto xi = torch::tensor({{0.2, 0.05}, {0.02, 0.3}}, torch::requires_grad());
  {
    torch::NoGradGuard no_grad;
    xi.mul_(trainable);
  }
  xi.sum().backward();
  eegexpcpp::xi_apply_trainable_mask(xi, trainable);
  DOCTEST_CHECK(xi[0][1].item<double>() == doctest::Approx(0.0));
  DOCTEST_CHECK(xi.grad()[0][1].item<double>() == doctest::Approx(0.0));
}

DOCTEST_TEST_CASE("sindy prune xi trainable buffer") {
  eegexpcpp::OdeConfig ocfg;
  ocfg.type = eegexpcpp::OdeType::Sindy;
  eegexpcpp::SindyLibraryConfig poly;
  poly.kind = eegexpcpp::SindyLibraryConfig::Kind::Poly;
  poly.poly.degree = 1;
  ocfg.libraries.push_back(poly);
  eegexpcpp::OdeSindy sindy(2, ocfg.libraries);
  const auto xi = torch::tensor({{0.2, 0.01}, {-0.15, 0.3}, {0.0, 0.5}});
  sindy->set_xi_matrix(xi);
  sindy->prune_xi(0.1);
  DOCTEST_CHECK(sindy->xi_nnz(0.1) == 4);
  DOCTEST_CHECK(sindy->xi_matrix()[0][1].item<double>() == doctest::Approx(0.0));
  for (const auto& pair : sindy->named_parameters()) {
    if (pair.key() == "xi") {
      DOCTEST_CHECK(pair.value()[0][1].item<double>() == doctest::Approx(0.0));
      break;
    }
  }
}

DOCTEST_TEST_CASE("sindy set xi matrix") {
  eegexpcpp::OdeConfig ocfg;
  ocfg.type = eegexpcpp::OdeType::Sindy;
  eegexpcpp::SindyLibraryConfig poly;
  poly.kind = eegexpcpp::SindyLibraryConfig::Kind::Poly;
  poly.poly.degree = 1;
  ocfg.libraries.push_back(poly);
  eegexpcpp::OdeSindy sindy(2, ocfg.libraries);
  auto xi = torch::ones({3, 2}) * 0.25;
  sindy->set_xi_matrix(xi);
  DOCTEST_CHECK(torch::allclose(sindy->xi_matrix(), xi));
}

DOCTEST_TEST_CASE("sindy coefficients table") {
  torch::manual_seed(7);
  eegexpcpp::ModelConfig mcfg;
  mcfg.n_latent = 2;
  mcfg.feature_dim = 3;
  mcfg.encoder_hidden = std::vector<int64_t>{4};
  mcfg.decoder_hidden = std::vector<int64_t>{4};
  mcfg.ode.type = eegexpcpp::OdeType::Sindy;
  eegexpcpp::SindyLibraryConfig poly;
  poly.kind = eegexpcpp::SindyLibraryConfig::Kind::Poly;
  poly.poly.degree = 1;
  mcfg.ode.libraries.push_back(poly);
  eegexpcpp::LatentAeOde model(mcfg);
  const auto table =
      eegexpcpp::build_sindy_coefficients_table(model, mcfg);
  DOCTEST_CHECK(table.features.size() == 3);
  DOCTEST_CHECK(table.latent_labels.size() == 2);
  DOCTEST_CHECK(table.xi.size() == 6);
}

DOCTEST_TEST_CASE("sindy enforce trainable mask after prune") {
  eegexpcpp::OdeConfig ocfg;
  ocfg.type = eegexpcpp::OdeType::Sindy;
  eegexpcpp::SindyLibraryConfig poly;
  poly.kind = eegexpcpp::SindyLibraryConfig::Kind::Poly;
  poly.poly.degree = 1;
  ocfg.libraries.push_back(poly);
  eegexpcpp::OdeSindy sindy(2, ocfg.libraries);
  const auto xi = torch::tensor({{0.2, 0.01}, {-0.15, 0.3}, {0.0, 0.5}});
  sindy->set_xi_matrix(xi);
  sindy->prune_xi(0.1);
  torch::Tensor xi_param;
  for (auto& pair : sindy->named_parameters()) {
    if (pair.key() == "xi") {
      xi_param = pair.value();
      break;
    }
  }
  DOCTEST_REQUIRE(xi_param.defined());
  {
    torch::NoGradGuard no_grad;
    xi_param[0][1] = 0.42;
  }
  sindy->enforce_trainable_mask();
  DOCTEST_REQUIRE(sindy->xi_matrix()[0][1].item<double>() == doctest::Approx(0.0));
  DOCTEST_REQUIRE(xi_param[0][1].item<double>() == doctest::Approx(0.0));
}

DOCTEST_TEST_CASE("xi prune due global epoch") {
  DOCTEST_CHECK_FALSE(eegexpcpp::xi_prune_due_epoch(9698, 3000));
  DOCTEST_CHECK(eegexpcpp::xi_prune_due_epoch(2999, 3000));
  DOCTEST_CHECK(eegexpcpp::xi_prune_due_epoch(8999, 3000));
  DOCTEST_CHECK_FALSE(eegexpcpp::xi_prune_due_epoch(9700, 3000));
  DOCTEST_CHECK_FALSE(eegexpcpp::xi_prune_due_epoch(199, 3000));
}

DOCTEST_TEST_CASE("config json") {
  const auto path = std::filesystem::path(__FILE__).parent_path().parent_path() /
                    "configs" / "latent_ae_ode.lorenz.json";
  const auto cfg = eegexpcpp::load_config(path.string());
  DOCTEST_CHECK(cfg.train.lr > 0.0);
  DOCTEST_CHECK(!cfg.data.h5_path.empty());
  DOCTEST_CHECK(cfg.model.n_latent == 3);
}
