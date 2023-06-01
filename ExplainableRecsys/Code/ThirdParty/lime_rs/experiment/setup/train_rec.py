import os
import pathlib

from Code.ThirdParty.lime_rs.experiment import utils
from Code.ThirdParty.lime_rs.src import data_utils
from Code.ThirdParty.lime_rs.src.model import FMRec

logger = utils.get_logger("train_rec")

working_dir = pathlib.Path(os.getcwd()).parent.parent.parent.parent
os.chdir(working_dir)


def main():
    print(f"Current path: {working_dir}")
    # setup
    exp_setup = utils.setup()

    # load
    logger.info("Load data")
    #dataset = data_utils.load_data()
    dataset = data_utils.loadDataV2()

    # train
    logger.info("Train FM model")
    rec_model = FMRec(rec_name=exp_setup.rec_name, dataset=dataset, uses_features=exp_setup.uses_features)
    rec_model.train()

    # write dump
    logger.info("Save rec model dump")
    data_utils.write_dump(rec_model, exp_setup.rec_name)


if __name__ == "__main__":
    main()
