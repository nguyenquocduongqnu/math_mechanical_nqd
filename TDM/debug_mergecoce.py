import logging
import os

import FrEIA.framework as Ff
import FrEIA.modules as Fm
import numpy as np
import ot
import pandas as pd
import torch
import torch.nn as nn
from scipy import optimize



def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else:  # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()


def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else:  # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum())


##################### MISSING DATA MECHANISMS #############################

##### Missing At Random ######

def MAR_mask(X, p, p_obs):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_obs = max(int(p_obs * d), 1)  ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = np.random.choice(d, d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Other variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    return mask


##### Missing not at random ######

def MNAR_mask_logistic(X, p, p_params=.3, exclude_inputs=True):
    """
    Missing not at random mechanism with a logistic masking model. It implements two mechanisms:
    (i) Missing probabilities are selected with a logistic model, taking all variables as inputs. Hence, values that are
    inputs can also be missing.
    (ii) Variables are split into a set of intputs for a logistic model, and a set whose missing probabilities are
    determined by the logistic model. Then inputs are then masked MCAR (hence, missing values from the second set will
    depend on masked values.
    In either case, weights are random and the intercept is selected to attain the desired proportion of missing values.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_params : float
        Proportion of variables that will be used for the logistic masking model (only if exclude_inputs).

    exclude_inputs : boolean, default=True
        True: mechanism (ii) is used, False: (i)

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_params = max(int(p_params * d), 1) if exclude_inputs else d  ## number of variables used as inputs (at least 1)
    d_na = d - d_params if exclude_inputs else d  ## number of variables masked with the logistic model

    ### Sample variables that will be parameters for the logistic regression:
    idxs_params = np.random.choice(d, d_params, replace=False) if exclude_inputs else np.arange(d)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_params]) if exclude_inputs else np.arange(d)

    ### Other variables will have NA proportions selected by a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_params, idxs_nas)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_params], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_params].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na)
    mask[:, idxs_nas] = ber < ps

    ## If the inputs of the logistic model are excluded from MNAR missingness,
    ## mask some values used in the logistic model at random.
    ## This makes the missingness of other variables potentially dependent on masked values

    if exclude_inputs:
        mask[:, idxs_params] = torch.rand(n, d_params) < p

    return mask


def MNAR_self_mask_logistic(X, p):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    ### Variables will have NA proportions that depend on those observed variables, through a logistic model
    ### The parameters of this logistic model are random.

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d) if to_torch else np.random.rand(n, d)
    mask = ber < ps if to_torch else ber < ps.numpy()

    return mask


def MNAR_mask_quantiles(X, p, q, p_params, cut='both', MCAR=False):
    """
    Missing not at random mechanism with quantile censorship. First, a subset of variables which will have missing
    variables is randomly selected. Then, missing values are generated on the q-quantiles at random. Since
    missingness depends on quantile information, it depends on masked values, hence this is a MNAR mechanism.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    q : float
        Quantile level at which the cuts should occur

    p_params : float
        Proportion of variables that will have missing values

    cut : 'both', 'upper' or 'lower', default = 'both'
        Where the cut should be applied. For instance, if q=0.25 and cut='upper', then missing values will be generated
        in the upper quartiles of selected variables.

    MCAR : bool, default = True
        If true, masks variables that were not selected for quantile censorship with a MCAR mechanism.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """
    n, d = X.shape

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    d_na = max(int(p_params * d), 1)  ## number of variables that will have NMAR values

    ### Sample variables that will have imps at the extremes
    idxs_na = np.random.choice(d, d_na, replace=False)  ### select at least one variable with missing values

    ### check if values are greater/smaller that corresponding quantiles
    if cut == 'upper':
        quants = quantile(X[:, idxs_na], 1 - q, dim=0)
        m = X[:, idxs_na] >= quants
    elif cut == 'lower':
        quants = quantile(X[:, idxs_na], q, dim=0)
        m = X[:, idxs_na] <= quants
    elif cut == 'both':
        u_quants = quantile(X[:, idxs_na], 1 - q, dim=0)
        l_quants = quantile(X[:, idxs_na], q, dim=0)
        m = (X[:, idxs_na] <= l_quants) | (X[:, idxs_na] >= u_quants)

    ### Hide some values exceeding quantiles
    ber = torch.rand(n, d_na)
    mask[:, idxs_na] = (ber < p) & m

    if MCAR:
        ## Add a mcar mecanism on top
        mask = mask | (torch.rand(n, d) < p)

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = torch.randn(d)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p

            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def generate_mask(X_true, missing_prop, mask_type=['MAR', 'MNARL', 'MNARQ', 'MCAR']):
    p_obs = 0.3  # Proportion of variables that are fully observed (MAR & MNAR model), set to 0.3 according to OTImputer
    q_mnar = 0.75  # Quantile that will have imps (MNAR quantiles model), set to 0.75 according to OTImputer
    if mask_type == 'MAR':
        mask = MAR_mask(X_true, missing_prop, p_obs).double()
    elif mask_type == 'MNARL':
        mask = MNAR_mask_logistic(X_true, missing_prop, p_obs).double()
    elif mask_type == "MNARQ":
        mask = MNAR_mask_quantiles(X_true, missing_prop, q_mnar, 1 - p_obs,
                                   cut='both', MCAR=False).double()
    else:
        mask = (torch.rand(X_true.shape) < missing_prop).double()

    return mask.cpu().numpy()

def run_TDM(X_missing, X_true=None):
    niter = 1000
    batchsize = 64
    lr = 1e-2
    report_interval = 100
    network_depth = 3
    network_width = 2


    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.DoubleTensor')

    # For small datasets, smaller batchsize may prevent overfitting;
    # For larger datasets, larger batchsize may give better performance.

    X_missing = torch.Tensor(X_missing)
    if X_true is not None:
        X_true = torch.Tensor(X_true)
    n, d = X_missing.shape
    mask = torch.isnan(X_missing)

    k = network_width

    def subnet_fc(dims_in, dims_out):
        return nn.Sequential(nn.Linear(dims_in, k * d), nn.SELU(), nn.Linear(k * d, k * d), nn.SELU(),
                             nn.Linear(k * d, dims_out))

    projector = Ff.SequenceINN(d)
    for _ in range(network_depth):
        projector.append(Fm.RNVPCouplingBlock, subnet_constructor=subnet_fc)

    imputer = TDM(projector, batchsize=batchsize, im_lr=lr, proj_lr=lr, niter=niter)
    imp, maes, rmses = imputer.fit_transform(X_missing.clone(), verbose=True, report_interval=report_interval,
                                             X_true=X_true)
    print(imp)
    print(rmses)

    #
    #
    # imp = imp.detach()
    #
    # result = {}
    # result["imp"] = imp[mask.bool()].detach().cpu().numpy()
    # if X_true is not None:
    #     result['learning_MAEs'] = maes
    #     result['learning_RMSEs'] = rmses
    #     result['MAE'] = MAE(imp, X_true, mask).item()
    #     result['RMSE'] = RMSE(imp, X_true, mask).item()
    #     OTLIM = 5000
    #     M = mask.sum(1) > 0
    #     nimp = M.sum().item()
    #     if nimp < OTLIM:
    #         M = mask.sum(1) > 0
    #         nimp = M.sum().item()
    #         dists = ((imp[M][:, None] - X_true[M]) ** 2).sum(2) / 2.
    #         result['OT'] = ot.emd2(np.ones(nimp) / nimp,
    #                                np.ones(nimp) / nimp, \
    #                                dists.cpu().numpy())
    #         logging.info(
    #             f"MAE: {result['MAE']:.4f}\t"
    #             f"RMSE: {result['RMSE']:.4f}\t"
    #             f"OT: {result['OT']:.4f}")
    #     else:
    #         logging.info(
    #             f"MAE: {result['MAE']:.4f}\t"
    #             f"RMSE: {result['RMSE']:.4f}\t")
    #

class TDM():

    def __init__(self,
                 projector,
                 im_lr=1e-2,
                 proj_lr=1e-2,
                 opt=torch.optim.RMSprop,
                 niter=2000,
                 batchsize=128,
                 n_pairs=1,
                 noise=0.1):

        self.im_lr = im_lr
        self.proj_lr = proj_lr
        self.opt = opt
        self.niter = niter
        self.batchsize = batchsize
        self.n_pairs = n_pairs
        self.noise = noise
        self.projector = projector

    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')
    else:
        torch.set_default_tensor_type('torch.DoubleTensor')

    def fit_transform(self, X, verbose=True, report_interval=500, X_true=None):

        X = X.clone()
        n, d = X.shape

        if self.batchsize > n // 2:
            e = int(np.log2(n // 2))
            self.batchsize = 2 ** e
            if verbose:
                logging.info(f"Batchsize larger that half size = {len(X) // 2}. Setting batchsize to {self.batchsize}.")

        mask = torch.isnan(X).double()

        torch.autograd.set_detect_anomaly(True)

        if torch.sum(mask) < 1.0:
            is_no_missing = True
        else:
            is_no_missing = False

        X_filled = X.detach().clone()

        if not is_no_missing:
            imps = (self.noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
            imps.requires_grad = True
            im_optimizer = self.opt([imps], lr=self.im_lr)
            X_filled[mask.bool()] = imps

        proj_optimizer = self.opt([p for p in self.projector.parameters()], lr=self.proj_lr)

        if X_true is not None:
            maes = np.zeros(self.niter)
            rmses = np.zeros(self.niter)

        for i in range(self.niter):

            X_filled = X.detach().clone()

            if not is_no_missing:
                X_filled[mask.bool()] = imps

            proj_loss = 0
            im_loss = 0

            for _ in range(self.n_pairs):
                idx1 = np.random.choice(n, self.batchsize, replace=False)
                idx2 = np.random.choice(n, self.batchsize, replace=False)

                X1 = X_filled[idx1]
                X2 = X_filled[idx2]

                X1_p, _ = self.projector(X1)
                X2_p, _ = self.projector(X2)

                M_p = torch.cdist(X1_p, X2_p, p=2)

                a1_p = torch.ones(X1.shape[0]) / X1.shape[0]
                a2_p = torch.ones(X2.shape[0]) / X2.shape[0]
                a1_p.requires_grad = False
                a2_p.requires_grad = False
                ot_p = ot.emd2(a1_p, a2_p, M_p)

                im_loss = im_loss + ot_p
                proj_loss = proj_loss + ot_p

            if torch.isnan(im_loss).any() or torch.isinf(im_loss).any():
                logging.info("im_loss Nan or inf loss")
                break

            if torch.isnan(proj_loss).any() or torch.isinf(proj_loss).any():
                logging.info("proj_loss Nan or inf loss")
                break

            if not is_no_missing:
                im_optimizer.zero_grad()
                im_loss.backward(retain_graph=True)
                im_optimizer.step()

            proj_optimizer.zero_grad()
            proj_loss.backward()
            proj_optimizer.step()

            if verbose and (i % report_interval == 0):

                if X_true is not None:
                    maes[i] = MAE(X_filled, X_true, mask).item()
                    rmses[i] = RMSE(X_filled, X_true, mask).item()

                    logging.info(f'Iteration {i}:\t Imputer Loss: {im_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t '
                                 f'Validation MAE: {maes[i]:.4f}\t'
                                 f'RMSE: {rmses[i]:.4f}')


                else:
                    logging.info(f'Iteration {i}:\t Imputer Loss: {im_loss.item():.4f}\t '
                                 f'Projector Loss: {proj_loss.item():.4f}\t ')

        X_filled = X.detach().clone()
        if not is_no_missing:
            X_filled[mask.bool()] = imps

        if X_true is not None:
            return X_filled, maes, rmses
        else:
            return X_filled



# For testing

missing_prop = 0.3
missing_type = 'MCAR' # Choosing from MAR, MNARL, MNARQ, MCAR
data = pd.read_csv(r'D:\Math_Mechanical\math_mechanical_nqd\Mechanical_Math_Application\data\test_dataset_prob_dl_no_null.csv').drop(['Unnamed: 0'], axis=1).drop(['Output'], axis=1)
df = data.to_numpy()
X_true = df
mask = generate_mask(X_true, missing_prop, missing_type)
X_missing = np.copy(X_true)
X_missing[mask.astype(bool)] = np.nan


run_TDM(X_missing, X_true)