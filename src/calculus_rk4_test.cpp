#include <cmath>
#include <vector>

#include <doctest.h>

#include "eegexpcpp/rk4.hpp"

TEST_CASE("custom_rk4 harmonic oscillator") {
  const double dt = 0.1;
  const int n = 63;
  const int ncol = 2;

  auto field = [](double* du, const double* u, double /*t*/) {
    du[0] = -u[1];
    du[1] = u[0];
  };

  const double u0[2] = {0.0, 1.0};
  auto out = eegexpcpp::custom_rk4(n, ncol, field, u0, dt);

  const double atol = 1e-4;
  for (int i = 0; i < n; ++i) {
    const double t = i * dt;
    const double ex0 = -std::sin(t);
    const double ex1 = std::cos(t);
    const double y0 = out[static_cast<size_t>(i * ncol)];
    const double y1 = out[static_cast<size_t>(i * ncol + 1)];
    CHECK(std::abs(y0 - ex0) <= atol);
    CHECK(std::abs(y1 - ex1) <= atol);
  }
}
