import pandas as pd
from nose.tools import assert_equal, assert_raises

from core.shots_analysis import ShotChartAnalysis


class TestShotChartAnalysis:

    def setup(self):
        self.shots_df = pd.DataFrame({
            'code': [
                '201412300POR', '201412300POR', '201412300POR',
                '201412300POR', '201412300POR', '201206210MIA',
                '201206210MIA', '201206210MIA', '201206210MIA',
                '201206210MIA'],
            'team': ['POR', 'POR', 'TOR', 'TOR', 'TOR',
                     'MIA', 'MIA', 'MIA', 'OKC', 'OKC'],
            'player_code': [
                'harrejo01', 'johnsmi01', 'tyrach01', 'hamilri01',
                'banksge01', 'delfica01', 'vanexni01', 'shirlpa01',
                'leeco01', 'vucevni01'],
            'x': [140, 334, 373, 3, 111, 188, 466, 187, 240, 229],
            'y': [31, 92, 184, 78, 28, 116, 33, 284, 31, 41]
        })
        self.shots = ShotChartAnalysis(self.shots_df)
        assert_equal(self.shots.hists, {})

    def test_shot_hist(self):
        hist = self.shots.shot_hist(target="hamilri01")
        assert_equal(hist[0].sum(), 1)
        assert_equal(hist[0].shape, (10, 10))
        hist = self.shots.shot_hist()
        assert_equal(hist[0].sum(), 10)
        hist = self.shots.shot_hist(target="hamilri01", bins=11)
        assert_equal(hist[0].shape, (11, 11))
        hist = self.shots.shot_hist(bins=11)
        assert_equal(hist[0].shape, (11, 11))
        hist = self.shots.shot_hist(group_col="code")
        assert_equal(hist[0].sum(), 10)
        assert_equal(hist[0].shape, (10, 10))
        hist = self.shots.shot_hist(group_col="code", target="201412300POR")
        assert_equal(hist[0].sum(), 5)
        hist = self.shots.shot_hist(group_col=["player_code"])
        assert_equal(hist[0].sum(), 10)
        hist = self.shots.shot_hist(group_col=["code", "team"])
        assert_raises(Warning, self.shots.shot_hist, target="alex")
        assert_raises(Exception, self.shots.shot_hist, group_col="foo")

    def test_calculate_hists(self):
        self.shots.calculate_hists()
        assert_equal(len(self.shots.hists), 10)
        self.shots.calculate_hists(group_col="code")
        assert_equal(len(self.shots.hists), 2)
        assert_raises(Exception, self.shots.shot_hist, group_col='foo')

    def test_hists_to_df(self):
        self.shots.calculate_hists()
        self.shots.hists_to_df()
        assert_equal(self.shots.hist_df.shape[0], 10)


