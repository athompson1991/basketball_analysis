import pandas as pd
import numpy as np

from nose.tools import assert_equal, assert_raises, assert_true
from numpy.testing import assert_equal as np_assert_equal

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
        np_assert_equal(hist[1], np.linspace(0, 500, num=11))
        np_assert_equal(hist[2], np.linspace(0, 450, num=11))

        hist = self.shots.shot_hist()
        assert_equal(hist[0].sum(), 10)
        assert_equal(hist[0].shape, (10, 10))

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

        hist = self.shots.shot_hist(group_col=["code", "team"],
                                    target="201412300POR/POR")
        assert_equal(hist[0].sum(), 2)

        assert_raises(Warning, self.shots.shot_hist, target="alex")
        assert_raises(Exception, self.shots.shot_hist, group_col="foo")

    def test_calculate_hists(self):
        self.shots.calculate_hists()
        assert_equal(len(self.shots.hists), 10)
        self.shots.calculate_hists(group_col="code")
        assert_equal(len(self.shots.hists), 2)
        self.shots.calculate_hists(group_col=["code", "team"])
        assert_equal(len(self.shots.hists), 4)
        assert_raises(Exception, self.shots.shot_hist, group_col='foo')
        assert_raises(Exception, self.shots.shot_hist, group_col=['foo', 'team'])

    def test_hists_to_df(self):
        l1 = self.shots_df['code'] == '201412300POR'
        l2 = self.shots_df['code'] == '201206210MIA'
        hist1 = np.histogram2d(self.shots_df[l1].x, self.shots_df[l1].y)
        hist2 = np.histogram2d(self.shots_df[l2].x, self.shots_df[l2].y)
        self.shots.hists['201412300POR'] = hist1
        self.shots.hists['201206210MIA'] = hist2

        self.shots.hists_to_df()
        assert_equal(self.shots.hist_df.shape[0], 2)
        assert_equal(self.shots.hist_df[0].dtype, np.int)
        self.shots.hists_to_df(freq=True)
        assert_equal(self.shots.hist_df.shape[0], 2)
        assert_equal(self.shots.hist_df[0].dtype, np.float)


