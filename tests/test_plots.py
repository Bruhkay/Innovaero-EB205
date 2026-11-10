"""Every figure renders from a (coarse) results dict without errors."""
import pytest

from cabinsim.pipeline import run
from cabinsim.plots import all_figures
from test_pipeline_smoke import _coarse


@pytest.fixture(scope="module")
def results(params):
    return run(_coarse(params), fast=True, log=lambda message: None)


def test_all_figures_render(results):
    figures = all_figures(results)
    assert "sil" in figures and "projection" in figures
    for figure in figures.values():
        figure.canvas.draw()
