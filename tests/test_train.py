import pytest
import hydra

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)

# Import train function
from src.train import train


@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path='../configs'):
        cfg = hydra.compose(
            config_name='train',
            overrides=['experiment=finetune'],
        )
        return cfg


def test_catdog_ex_training(config, tmp_path):
    # Update output and log directories to use temporary path
    config.paths.output_dir = str(tmp_path)
    config.paths.log_dir = str(tmp_path / 'logs')

    # Instantiate components
    datamodule = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)
    trainer = hydra.utils.instantiate(config.trainer)

    # Run training
    train(config, trainer, model, datamodule)
