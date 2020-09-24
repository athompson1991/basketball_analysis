import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ShotChartAnalysis:

    def __init__(self, shots):
        self.shots = shots.copy()
        self.hists = {}
        self.hist_df = None
        self.pca_df = None

    def calculate_hists(self, group_col='player_code', **kwargs):
        if group_col not in self.shots.columns:
            raise ValueError("Cannot find grouping in dataframe")
        self.hists = {}
        for target in self.shots[group_col].unique():
            try:
                self.hists[target] = self.shot_hist(
                    group_col=group_col, target=target, **kwargs)
                print("Completed hist: " + target)
            except Exception as e:
                raise Warning(f"Missing: {target}")

    def shot_hist(self, group_col="player_code", target=None, **kwargs):
        df = self.shots
        if isinstance(group_col, str):
            group_col = [group_col]
        for group in group_col:
            if group not in df.columns:
                raise ValueError(f"Cannot find column in dataframe: {group}")
        if target is not None:
            if target not in df[group_col].values:
                raise Warning(f"Not in values: {target}")
            df_ls = df.loc[:, group_col].values.tolist()
            lookup = pd.Series(map("/".join, df_ls))
            logic = lookup == target
            df = self.shots.loc[logic]
        hist = np.histogram2d(df['x'], df['y'], **kwargs)
        return hist

    def hists_to_df(self):
        df_dict = {}
        for player in self.hists.keys():
            n = self.hists[player][0].sum()
            df_dict[player] = self.hists[player][0] / n
            df_dict[player] = df_dict[player].flatten()
        self.hist_df = pd.DataFrame(df_dict).transpose()

    def do_pca(self, pct=0.9):
        df_scaled = StandardScaler().fit_transform(self.hist_df)
        pca = PCA(n_components=pct)
        self.pca_df = pca.fit_transform(df_scaled)
        self.pca_df = pd.DataFrame(self.pca_df)
        self.pca_df.set_index(self.hist_df.index, inplace=True)

    def hist_plot(self, player, ax=plt.subplot()):
        hist = self.hists[player][0]
        hist = hist / hist.sum()
        ax.title.set_text(player)
        ax.imshow(hist)
