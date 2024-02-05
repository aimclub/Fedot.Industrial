from fedot_ind.core.architecture.pipelines.abstract_pipeline import AbstractPipelines
from fedot_ind.tools.synthetic.ts_datasets_generator import TimeSeriesDatasetsGenerator


def test_ppl_setup():
    train_data, test_data = TimeSeriesDatasetsGenerator().generate_data()
    ppl = AbstractPipelines(train_data, test_data)
    assert ppl.test_features is not None
    assert ppl.train_features is not None
    assert ppl.test_target is not None
    assert ppl.train_target is not None
