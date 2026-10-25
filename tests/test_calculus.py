import numpy as np

from eegexpcpp.calculus import custom_rk4, diff4_order1, integ4_order1


def _t_grid():
    return np.arange(0, 2 * np.pi, 0.1, dtype=np.float64)


class TestDifferentiation:
    def test_sin_cos_full(self):
        t = _t_grid()
        dt = 0.1
        np.testing.assert_allclose(
            diff4_order1(np.sin(t), dt), np.cos(t), atol=0.01
        )
        np.testing.assert_allclose(
            diff4_order1(np.cos(t), dt), -np.sin(t), atol=0.01
        )

    def test_sin_cos_interior(self):
        t = _t_grid()
        dt = 0.1
        np.testing.assert_allclose(
            diff4_order1(np.sin(t), dt)[2:-2],
            np.cos(t)[2:-2],
            atol=0.0001,
        )
        np.testing.assert_allclose(
            diff4_order1(np.cos(t), dt)[2:-2],
            -np.sin(t)[2:-2],
            atol=0.0001,
        )

    def test_two_columns_dim1(self):
        t = _t_grid()
        dt = 0.1
        x = np.column_stack([np.sin(t), np.cos(t)])
        expected = np.column_stack([np.cos(t), -np.sin(t)])
        np.testing.assert_allclose(
            diff4_order1(x, dt, dim=1), expected, atol=0.01
        )


class TestIntegration:
    def test_sin_cos(self):
        t = _t_grid()
        dt = 0.1
        np.testing.assert_allclose(
            integ4_order1(np.sin(t), dt, np.array([-1.0])),
            -np.cos(t),
            atol=0.0001,
        )
        np.testing.assert_allclose(
            integ4_order1(np.cos(t), dt, np.array([0.0])),
            np.sin(t),
            atol=0.0001,
        )

    def test_two_columns_dim1(self):
        t = _t_grid()
        dt = 0.1
        x = np.column_stack([np.sin(t), np.cos(t)])
        expected = np.column_stack([-np.cos(t), np.sin(t)])
        np.testing.assert_allclose(
            integ4_order1(x, dt, np.array([[-1.0, 0.0]]), dim=1),
            expected,
            atol=0.0001,
        )


class TestCustomRk4:
    def test_harmonic_oscillator(self):
        def simple_du(du, u, p, t):
            du[0] = -u[1]
            du[1] = u[0]

        t = _t_grid()
        dt = 0.1
        n = len(t)
        u0 = np.array([[0.0, 1.0]])
        out = custom_rk4(simple_du, u0, None, dt, n)
        expected = np.column_stack([-np.sin(t), np.cos(t)])
        np.testing.assert_allclose(out, expected, atol=0.0001)
