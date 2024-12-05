""" Holds the data needed for plotting """
import torch
import numpy as np
import os.path as op
import os
import sys
from epviz import models

class PredictionInfo():
    """ Data structure for holding model and preprocessed data for prediction """
    def __init__(self, parent):
        self.parent = parent
        self.ready = 0 # whether both model and data have been loaded
        self.model = [] # loaded model
        self.is_dl_model = False
        self.data = [] # loaded data for model
        self.model_preds = [] # predictions from model/data
        self.preds = [] # loaded predictions
        self.preds_to_plot = [] # predictions that are to be plotted
                                # preds should be of shape [n_preds, nchns] if multi-chn
        self.dl_model = None # the deep learning model
        self.dl_model_params = None # the parameters for the deep learning model
        self.model_fn = "" # name of the model file
        self.data_fn = "" # name of the data file
        self.preds_fn = "" # name of the loaded predictions file
        self.model_loaded = 0 # if the model has been loaded
        self.data_loaded = 0 # if the data has been loaded
        self.preds_loaded = 0 # if the predictions have been loaded
        self.multi_class_model = 0 # currently plotting multiclass from model
        self.multi_class_preds = 0 # currently plotting mutliclass from loaded preds
        self.plot_model_preds = 0 # whether or not to load model_preds into preds_to_plot
        self.plot_loaded_preds = 0 # whether or not to load preds into preds_to_plot
        self.pred_width = 0 # width in samples of each prediction, must be an int
        self.pred_by_chn = 0 # whether or not we are predicting by channel
        self.multi_class = 0 # whether or not we are doing multi-class predictions
        self.predicted = 0
        # default colors to use for multi-class:
        # transparent, dark blue, green, yellow, red, pink, purple
        self.class_colors = [(255,255,255,0),(50, 95, 168, 50), (50, 168, 82,50), (168, 52, 50, 50),
                                (164, 168, 50, 50), (168, 50, 150,50), (127, 50, 168, 50)]
    def write_data(self, pi2):
        """ Writes data from pi2 into self.
        """
        self.ready = pi2.ready
        self.model = pi2.model
        self.data = pi2.data
        self.model_preds = pi2.model_preds
        self.preds = pi2.preds
        self.preds_to_plot = pi2.preds_to_plot
        self.model_fn = pi2.model_fn
        self.data_fn = pi2.data_fn
        self.preds_fn = pi2.preds_fn
        self.model_loaded = pi2.model_loaded
        self.data_loaded = pi2.data_loaded
        self.preds_loaded = pi2.preds_loaded
        self.plot_model_preds = pi2.plot_model_preds
        self.plot_loaded_preds = pi2.plot_loaded_preds
        self.pred_width = pi2.pred_width
        self.pred_by_chn = pi2.pred_by_chn

    def get_color(self, i):
        """ Returns a color for multi-class prediction

            Args:
                i - the index of the color
            Returns:
                A color in (R, G, B, alpha) form (alpha is set to 50)
        """
        if i < 6:
            return self.class_colors[i]
        num0 = 255
        num1 = 255
        num2 = 255
        while (num0, num1, num2, 0) in self.class_colors:
            num0 = np.random.random() * 255
            num1 = np.random.random() * 255
            num2 = np.random.random() * 255
        return (num0, num1, num2, 50)

    def set_data(self, data_fn):
        """ Set the data to the data from data_fn
        """
        if data_fn.endswith(".npy"):
            self.data = np.load(data_fn)
        else:
            self.data = torch.load(data_fn)
        self.data_loaded = 1
        self.update_ready()

    def set_model(self, model_fn):
        """ Load in the model given by model_fn.
        """
        if model_fn.endswith(".skops"):
            from skops.io import load
            self.model = load(model_fn)
        else:
            self.model = torch.load(model_fn)
        self.model_loaded = 1
        self.update_ready()

    def update_ready(self):
        """ Updates whether the data and model are loaded.
        """
        if self.model_loaded and self.data_loaded:
            self.ready = 1

    def set_preds(self, preds_fn, max_time, fs, nchns, binary = True):
        """ Loads predictions.

        Returns:
            0 for sucess, -1 if predictions are not the right length
            predictions must be for an integer number of samples in the file
        """
        try:
            if preds_fn.endswith(".npy"):
                preds = np.load(preds_fn)
            else:
                preds = torch.load(preds_fn)
        except:
            raise Exception("The predictions file could not be loaded.")

        if not isinstance(preds, np.ndarray):
            try:
                preds = preds.detach()
            except:
                pass
            preds = preds.numpy()
        ret = self.check_preds_shape(preds, 0, max_time, fs, nchns, binary)
        return ret

    def predict(self, max_time, fs, nchns, binary = True):
        """ Loads model, passes data through the model to get
            seizure predictions

        Args:
            data - the pytorch tensor, fully preprocessed
            model_fn - filename of the model to load
            binary - whether to do binary or multi-class

        Returns:
            0 for sucess, -2 for failure to pass through the predict function,
            -1 for incorrect size
        """
        try:
            preds = self.model.predict(self.data)
            preds = np.array(preds)
        except:
            return -2

        ret = self.check_preds_shape(preds, 1, max_time, fs, nchns, binary)
        return ret

    def run_dl_model(self):
        """ Runs the deep learning model on the data"""

        if self.dl_model is None or self.dl_model_params is None:
            return -1

        epviz_dir = op.dirname(op.dirname(op.dirname(op.abspath(__file__))))
        deepsoz_path = op.join(epviz_dir, "dl_models", "DeepSOZ")
        deepsoz_code_dir = op.join(deepsoz_path, "code", "test")
        deepsoz_model_params_dir = op.join(deepsoz_path, "final_models")

        if deepsoz_code_dir not in sys.path:
            sys.path.append(deepsoz_code_dir)

        # Get params
        input_model_params = self.dl_model_params
        self.dl_model_params = ""
        self.dl_model_params = op.join(deepsoz_model_params_dir, *input_model_params.split(os.sep)[1:])
        params = torch.load(self.dl_model_params, map_location=torch.device('cpu'))

        # Set the model
        if self.dl_model == "txlstm_szpool":
            from txlstm_szpool import txlstm_szpool
            model = txlstm_szpool()
            model.load_state_dict(params)
        elif self.dl_model == "cnn_blstm":
            from baselines import CNN_BLSTM
            model = CNN_BLSTM()
            model.load_state_dict(params)

        # Format data properly: Resample to 200Hz and add zero padding if needed
        edf_info = self.parent.edf_info
        n_samples = edf_info.nsamples[0]
        fs = int(edf_info.fs)

        # Get data
        start_idx = self.parent.count * int(fs)
        stop_idx = (self.parent.count + 600) * int(fs)
        if stop_idx > n_samples:
            self.parent.throw_alert("Not enough data available! Set current time earlier to run the model.")
            return

        # Check if montage is average ref 1020
        chns = self.parent.ci.labels_to_plot
        if "Notes" in chns:
            chns.pop(chns.index("Notes"))

        ar1020_chns = self.parent.ci.labelsAR1020
        bool_chns = [False] * len(chns)
        for i, chn in enumerate(chns):
            if chn in ar1020_chns:
                bool_chns[i] = True
        if not all(bool_chns):
            self.parent.throw_alert("Model can only be run on 1020 average reference montage.")
            return

        # Finally retrieve the data
        input_data = self.parent.ci.get_data(
            self.parent.ci.list_of_chns,
            self.parent,
            self.parent.ci.mont_type,
            start_idx=start_idx,
            stop_idx=stop_idx)

        # Resample
        from scipy.signal import resample, resample_poly
        new_fs = 200
        new_num_samples = int(input_data.shape[1] * fs / new_fs)
        #input_data = resample(input_data, new_num_samples, axis=1)
        input_data = resample_poly(input_data, new_fs, fs, axis=1)

        # Make into torch.tensor datatype to input into model
        reshaped_data = input_data.reshape(input_data.shape[0], 600, 200)  # Shape: (19, 600, 200)
        reshaped_data = np.transpose(reshaped_data, (1, 0, 2))  # Shape: (600, 19, 200)
        input_data_tensor = torch.tensor(reshaped_data).unsqueeze(0).unsqueeze(0)

        # Run model
        #model = model.double()
        model_output = model(input_data_tensor.float())
        model_output = torch.nn.functional.softmax(model_output[0], dim=-1)
        model_output = model_output[..., 1]
        model_output = model_output.squeeze().detach().numpy()

        # Reshape to match data. Add zero padding to match original data size
        n_samples_per_second = fs
        model_output = model_output.repeat(n_samples_per_second)
        output = np.zeros(n_samples)
        output[start_idx:stop_idx] = model_output[:stop_idx - start_idx]

        self.pred_width = 1

        return output


    def check_preds_shape(self, preds, model_or_preds, max_time, fs, nchns, binary = True):
        """
        Checks whether the predictions are the proper size.
        Samples in the file must be an integer multiple of length.
        The other dimension must be either 1, 2 (which will be collapsed) or nchns
        size: (num predictions, <chns, optional>, num classes)

        Args:
            preds - the predictions (np array)
            model_or_preds - 1 for model loaded, 0 for loaded predictions
            max_time - amount of seconds in the .edf file
            nchns - number of channels
            binary - whether to do binary or multi-class
        Returns:
            0 for sucess, -1 for incorrect length
        """
        # check size
        ret = -1
        self.pred_by_chn = 0 # reset
        self.multi_class = 0 # reset
        if binary:
            if len(preds.shape) == 3:
                # really should be in axis=2
                # but we'll check them all to be sure
                if preds.shape[2] == 2:
                    preds = preds[:,:,1]
                elif preds.shape[1] == 2:
                    preds = preds[:,1,:]
                elif preds.shape[0] == 2:
                    preds = preds[1,:,:]
            preds = np.squeeze(preds)
            dim = len(preds.shape)
            if dim == 1:
                if (fs * max_time) % preds.shape[0] == 0:
                    ret = 0
            elif dim == 2:
                if (fs * max_time) % preds.shape[0] == 0:
                    if preds.shape[1] <= 2:
                        ret = 0
                        if preds.shape[1] == 2:
                            preds = preds[:,1]
                    elif preds.shape[1] == nchns:
                        self.pred_by_chn = 1
                        ret = 0
        else:
            if (fs * max_time) % preds.shape[0] == 0:
                if len(preds.shape) == 3:
                    if preds.shape[1] != nchns:
                        return -1
                    self.pred_by_chn = 1
                ret = 0
                self.multi_class = 1

        if ret == 0:
            self.pred_width = fs * max_time / preds.shape[0]
            if model_or_preds: # model
                self.model_preds = preds
            else: # loaded_preds
                self.preds_loaded = 1
                self.preds = preds
        return ret

    def compute_starts_ends_chns(self, thresh, count, ws, fs, nchns):
        """
        Computes start / end / chn values of predictions in given window in samples

        Input:
            thresh - the threshold
            count - the current time in seconds
            ws - the current window_size in seconds
            fs - the frequency
        Output:
            starts - the start times
            ends - the corresponding end times
            chns - the channel to plot the given prediction (if
                    multi-class and multi-channel, class preds
                    will be stored here instead of in chns)
            class_vals - the values for each class
        """
        start_t = count * fs
        end_t = start_t + ws * fs
        starts = []
        ends = []
        chns = []
        class_vals = []
        pw = self.pred_width

        if self.pred_by_chn and not self.multi_class:
            preds_flipped = np.zeros(self.preds_to_plot.shape)
            for i in range(preds_flipped.shape[1]):
                preds_flipped[:,i] += self.preds_to_plot[:,self.preds_to_plot.shape[1] - i - 1]

            preds_mutli_chn = np.zeros((self.preds_to_plot.shape[0], nchns))
            if self.preds_to_plot.shape[1] >= nchns:
                preds_mutli_chn += preds_flipped[:,self.preds_to_plot.shape[1] - nchns:]
            else:
                preds_mutli_chn[:,nchns - self.preds_to_plot.shape[1]:] += preds_flipped
        if self.pred_by_chn and self.multi_class:
            preds_flipped = np.zeros(self.preds_to_plot.shape)
            for i in range(preds_flipped.shape[1]):
                preds_flipped[:,i,:] += self.preds_to_plot[:,self.preds_to_plot.shape[1] - i - 1,:]

            preds_mutli_chn_multi_class = np.zeros((self.preds_to_plot.shape[0],
                        nchns, self.preds_to_plot.shape[2]))
            if self.preds_to_plot.shape[1] >= nchns:
                preds_mutli_chn_multi_class += (
                    preds_flipped[:,self.preds_to_plot.shape[1] - nchns:,:])
            else:
                preds_mutli_chn[:,nchns - self.preds_to_plot.shape[1]:,:] += preds_flipped

        i = 0
        while i * pw < start_t: # find starting value
            i += 1
        i -= 1
        if i * pw < start_t < (i + 1) * pw:
            if self.multi_class:
                starts.append(start_t)
                ends.append((i + 1) * pw)
                val = np.argmax(self.preds_to_plot[i])
                class_vals.append(val)
                if self.pred_by_chn:
                    val = np.argmax(self.preds_to_plot[i], axis=1)
                    chns.append(val)
            else:
                if np.max(self.preds_to_plot[i]) > thresh:
                    starts.append(start_t)
                    ends.append((i + 1) * pw)
                    if self.pred_by_chn:
                        chn_i = preds_mutli_chn[i] > thresh
                        chns.append(chn_i)
            i += 1
        while i * pw < start_t: # find starting value
            i += 1
        while i * pw < end_t:
            if (i + 1) * pw > end_t:
                if self.multi_class:
                    starts.append(i * pw)
                    ends.append((i + 1) * pw)
                    val = np.argmax(self.preds_to_plot[i])
                    class_vals.append(val)
                    if self.pred_by_chn:
                        val = np.argmax(self.preds_to_plot[i],axis=1)
                        chns.append(val)
                    return starts, ends, chns, class_vals
                else:
                    if np.max(self.preds_to_plot[i]) > thresh:
                        starts.append(i * pw)
                        ends.append(end_t)
                        if self.pred_by_chn:
                            chn_i = preds_mutli_chn[i] > thresh
                            chns.append(chn_i)
                        return starts, ends, chns, class_vals
            if self.multi_class:
                starts.append(i * pw)
                ends.append((i + 1) * pw)
                val = np.argmax(self.preds_to_plot[i])
                class_vals.append(val)
                if self.pred_by_chn:
                    val = np.argmax(self.preds_to_plot[i],axis=1)
                    chns.append(val)
            elif np.max(self.preds_to_plot[i]) > thresh:
                starts.append(i * pw)
                ends.append((i + 1) * pw)
                if self.pred_by_chn:
                    chn_i = preds_mutli_chn[i] > thresh
                    chns.append(chn_i)
            i += 1
        return starts, ends, chns, class_vals
