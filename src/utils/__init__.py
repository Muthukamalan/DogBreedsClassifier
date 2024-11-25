# from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
# from src.utils.logging_utils import log_hyperparameters
# from src.utils.pylogger import RankedLogger
# from src.utils.rich_utils import enforce_tags, print_config_tree
# from src.utils.utils import extras, get_metric_value, task_wrapper

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


from src.utils.logging_utils import (
    setup_logger,
    task_wrapper,
    get_rich_progress,
    plot_confusion_matrix,
)