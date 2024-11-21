import pytest
import hydra

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(config_name='eval', return_hydra_config=True)
        return cfg


def test_catdog_ex_testing(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / 'logs')

    # Instantiate components
    datamodule = hydra.utils.instantiate(config.data)
    datamodule.setup()
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer)

    trainer.test(model, datamodule=datamodule, verbose=True, ckpt_path=config.ckpt_path)
