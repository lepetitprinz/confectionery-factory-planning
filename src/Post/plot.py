import common.util as util

import os
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


class Gantt(object):
    def __init__(self, fp_version, fp_seq, plant, path):
        self.fp_version = fp_version
        self.fp_seq = fp_seq
        self.plant = plant
        self.path = path

    def draw(self, data, y, color, name):
        fig = self.plot(data=data, y=y, color=color)
        self.save(fig=fig, name=name)

    def plot(self, data: pd.DataFrame, y: str, color: str):
        fig = px.timeline(
            data,
            x_start='start',
            x_end='end',
            y=y,
            color=color,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(autorange="reversed")

        return fig

    def save(self, fig, name):
        save_dir = os.path.join(self.path, 'gantt', self.fp_version)
        util.make_dir(path=save_dir)

        fig.write_html(os.path.join(save_dir, name + '_' + self.fp_seq + '_' + self.plant + '.html'))

        plt.close()