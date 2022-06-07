from core.operation.utils.reporter import TabularReporter


Reporter = TabularReporter()
df_report, generation_dict, metric_dict = Reporter.create_report(launches=5,
                                                                 save_flag=True)
_ = 1
