from eegexpcpp.sindy_table import parse_sindy_coefficients_csv


def test_parse_sindy_csv() -> None:
    text = "feature,dz0,dz1\n1,0.5,-0.1\nx0,1.0,0.0\n"
    table = parse_sindy_coefficients_csv(text)
    assert table["feature"] == ["1", "x0"]
    assert table["dz0"] == [0.5, 1.0]
    assert table["dz1"] == [-0.1, 0.0]
