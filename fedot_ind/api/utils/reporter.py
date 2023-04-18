import os

import pandas as pd


class ReporterTSC:
    def __init__(self, path_to_save: str = None):
        self._report = []
        self.path_to_save = path_to_save

    @property
    def report(self):
        return self._report

    def add_to_report(self, dataset_name, generator, launch_num, result):
        self._report.append({'dataset': dataset_name,
                             'generator': generator,
                             'launch_num': launch_num,
                             'result': result})

    def get_summary(self):
        report_df = pd.DataFrame(self.report)
        report_path = os.path.join(self.path_to_save, 'summary.csv')
        report_df.to_csv(report_path, index=False)
        return report_df


if __name__ == '__main__':
    reporter = ReporterTSC()
    reporter.add('beef', 'quantile', 1, 'good')
    reporter.add('beef', 'quantile', 2, 'bad')
    reporter.add('beef', 'wavelet', 1, 'good')
    reporter.add('beef', 'wavelet', 2, 'bad')
    reporter.add('lightning', 'quantile', 1, 'good')
    reporter.add('lightning', 'quantile', 2, 'bad')

    print(reporter.get('dataset_name'))
