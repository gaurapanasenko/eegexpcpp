#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace eegexpcpp {

/// Row-major ``u`` with shape ``[n, ncol]``. ``u0`` is the first row
/// (``ncol`` elements). ``f`` writes time derivative into ``du``.
template <typename T, typename F>
void custom_rk4_inplace(T* u,
                        int64_t n,
                        int64_t ncol,
                        F&& f,
                        const T* u0,
                        T dt) {
  if (ncol < 1) {
    throw std::invalid_argument("ncol must be >= 1");
  }
  if (n < 1) {
    throw std::invalid_argument("n must be >= 1");
  }
  for (int64_t c = 0; c < ncol; ++c) {
    u[c] = u0[c];
  }
  if (n == 1) {
    return;
  }

  std::vector<T> buf(static_cast<size_t>(7 * ncol));
  T* k1 = buf.data();
  T* k2 = k1 + ncol;
  T* k3 = k2 + ncol;
  T* k4 = k3 + ncol;
  T* tmp1 = k4 + ncol;
  T* tmp2 = tmp1 + ncol;
  T* tmp3 = tmp2 + ncol;

  const T half = T(0.5);
  const T one = T(1);
  const T sixth = dt / T(6);

  for (int64_t i = 0; i < n - 1; ++i) {
    const double ti = static_cast<double>(i) * static_cast<double>(dt);
    const T* ui = u + i * ncol;
    T* u_next = u + (i + 1) * ncol;

    f(k1, ui, ti);
    for (int64_t c = 0; c < ncol; ++c) {
      tmp1[c] = ui[c] + half * dt * k1[c];
    }
    f(k2, tmp1, ti + 0.5 * static_cast<double>(dt));
    for (int64_t c = 0; c < ncol; ++c) {
      tmp2[c] = ui[c] + half * dt * k2[c];
    }
    f(k3, tmp2, ti + 0.5 * static_cast<double>(dt));
    for (int64_t c = 0; c < ncol; ++c) {
      tmp3[c] = ui[c] + one * dt * k3[c];
    }
    f(k4, tmp3, ti + static_cast<double>(dt));
    for (int64_t c = 0; c < ncol; ++c) {
      u_next[c] =
          ui[c] + sixth * (k1[c] + T(2) * k2[c] + T(2) * k3[c] + k4[c]);
    }
  }
}

template <typename T, typename F>
std::vector<T> custom_rk4(int64_t n,
                          int64_t ncol,
                          F&& f,
                          const T* u0,
                          T dt) {
  std::vector<T> out(static_cast<size_t>(n * ncol),
                       std::numeric_limits<T>::quiet_NaN());
  custom_rk4_inplace(out.data(), n, ncol, std::forward<F>(f), u0, dt);
  return out;
}

}  // namespace eegexpcpp
