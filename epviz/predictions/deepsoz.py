import sys
import os.path as op
import glob
import torch

#epviz_dir = "/"
epviz_dir = op.dirname(op.dirname(op.abspath(__file__)))


# Append DeepSOZ path to sys.path
# deepsoz_path = op.join(epviz_dir, "dl_models", "DeepSOZ")
# deeepsoz_code_dir = op.join(epviz_dir, "code", "test")
# deepsoz_model_params_dir = op.join(deepsoz_path, "final_models")
# sys.path.append(deepsoz_path)
#
# # Search for DeepSOZ model files
#
#
# # %% DeepSOZ: txlstm_szpool
# import txlstm_szpool
#
# # Search for model files
# model_files = sorted(glob.glob(op.join(deepsoz_model_params_dir, "*", "txlstm_szpool*")))
#
# # Choosen a model file
# parameters_fname = model_files[0]
#
# # Load model
# parameters = torch.load(parameters_fname, map_location=torch.device('cpu'))
# deepsoz = txlstm_szpool.txlstm_szpool()
# deepsoz.load_state_dict(parameters)
#
# # %% DeepSOZ: CNN_BLSTM
# from baselines import CNN_BLSTM
#
# # Search for model files
# model_files = sorted(glob.glob(op.join(deepsoz_model_params_dir, "*", "cnnblstm*")))
#
# # Choosen a model file
# parameters_fname = model_files[0]
#
# # Load model
# parameters = torch.load(parameters_fname, map_location=torch.device('cpu'))
# cnn_blstm_model = CNN_BLSTM()
# cnn_blstm_model.load_state_dict(parameters)

def get_deepsoz_models_and_params():
    """
    Get the available DeepSOZ models and their parameters.
    """
    epviz_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
    deepsoz_path = op.join(epviz_dir, "dl_models", "DeepSOZ")
    deepsoz_model_params_dir = op.join(deepsoz_path, "final_models")

    # Get Transformed LSTM model params options
    txlstm_model_files = sorted(glob.glob(op.join(deepsoz_model_params_dir, "*", "txlstm_szpool*")))
    txlstm_model_files = [f.replace(deepsoz_model_params_dir, "") for f in txlstm_model_files]

    # Get CNN_BLSTM model params options
    cnn_blstm_model_files = sorted(glob.glob(op.join(deepsoz_model_params_dir, "*", "cnnblstm*")))
    cnn_blstm_model_files = [f.replace(deepsoz_model_params_dir, "") for f in cnn_blstm_model_files]

    return {
        "txlstm_szpool": txlstm_model_files,
        "cnn_blstm": cnn_blstm_model_files
    }