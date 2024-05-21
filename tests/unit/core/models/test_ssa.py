

def test_ssa():
    pass
    # time_series = np.random.normal(size=30)
    # task = Task(TaskTypesEnum.ts_forecasting,
    #             TsForecastingParams(forecast_length=1))
    # train_input = InputData(idx=np.arange(time_series.shape[0]),
    #                         features=time_series,
    #                         target=time_series,
    #                         task=task,
    #                         data_type=DataTypesEnum.ts)
    # train_data, test_data = train_test_data_setup(train_input)
    #
    # with IndustrialModels():
    #     pipeline = PipelineBuilder().add_node('ssa_forecaster').build()
    #     pipeline.fit(train_data)
    # assert pipeline is not None
