from math import floor

import numpy as np
from custom_utilities_methods import *
from scipy.sparse import csr_matrix
from tqdm import tqdm
from wrappers_threshold_methods import *

from xcolumns.block_coordinate import (
    _bc_with_0approx_step_csr,
    _bc_with_0approx_step_dense,
)
from xcolumns.confusion_matrix import _update_unnormalized_confusion_matrix
from xcolumns.frank_wolfe import (
    _metric_func_with_gradient_autograd,
    _metric_func_with_gradient_torch,
    find_classifier_using_fw,
)
from xcolumns.metrics import *
from xcolumns.types import *
from xcolumns.utils import *
from xcolumns.weighted_prediction import *


DEFAULT_REG = 1e-6


class OnlineMethod:
    def __init__(self, m, k):
        self.m = m
        self.k = k

    def predict_online(self, y_proba, y_pred, i):
        pass

    def update_online(self, y_true, y_pred, y_proba, i):
        pass

    def predict_batch(self, y_proba, y_pred, batch):
        for i in batch:
            self.predict_online(y_proba, y_pred, i)

    def has_pu_solution(self):
        return True

    def get_meta(self):
        return {}


class OnlineThresholds(OnlineMethod):
    def __init__(self, m, k, binary_utility_func, update_base=10, update_exp=1.5):
        super().__init__(m, k)
        self.binary_utility_func = binary_utility_func

        self.thresholds = np.full(m, 0.5, dtype=FLOAT_TYPE)
        self.seen_so_far = []
        self.update_base = update_base
        self.update_exp = update_exp
        self.updates = []

    def predict_online(self, y_proba, y_pred, i):
        y_pred[i, :] = 0
        if self.k > 0:
            gains = y_proba[i] - self.thresholds
            top_k = np.argpartition(-gains, self.k)[: self.k]
            y_pred[i, top_k] = 1.0
        else:
            y_pred[i, y_proba[i] > self.thresholds] = 1.0

    def update_online(self, y_true, y_pred, y_proba, i):
        self.seen_so_far.append(i)
        if len(self.seen_so_far) % self.update_base == self.update_base - 1:
            self.thresholds, _ = find_thresholds(
                y_true[self.seen_so_far, :],
                y_proba[self.seen_so_far, :],
                self.binary_utility_func,
            )
            self.update_base = floor(self.update_base * self.update_exp)
            self.updates.append(len(self.seen_so_far))

    def get_meta(self):
        return {"thresholds": self.thresholds, "updates_steps": self.updates}


class OnlineFrankWolfe(OnlineMethod):
    def __init__(
        self,
        m,
        k,
        utility_func,
        first_update_n=0,
        update_base=10,
        update_exp=1.1,
        skip_tn=False,
        etu_variant=False,
    ):
        super().__init__(m, k)
        self.utility_func = utility_func
        self.skip_tn = skip_tn
        self.etu_variant = etu_variant

        self.classifiers_a = np.zeros((1, m), dtype=FLOAT_TYPE)
        self.classifiers_b = np.zeros((1, m), dtype=FLOAT_TYPE)
        self.classifiers_proba = np.ones(1, dtype=FLOAT_TYPE)

        self.classifiers_a[0] = np.ones(m, dtype=FLOAT_TYPE)
        self.classifiers_b[0] = np.full(m, -0.5, dtype=FLOAT_TYPE)

        self.seen_so_far = []
        self.update_base = update_base
        self.update_exp = update_exp
        self.next_update = update_base
        self.first_update_n = first_update_n
        self.updates = []

    def predict_online(self, y_proba, y_pred, i):
        if isinstance(y_proba, csr_matrix):
            c = np.random.choice(
                np.arange(self.classifiers_proba.size), p=self.classifiers_proba
            )
            (
                y_pred.data,
                y_pred.indices,
                y_pred.indptr,
            ) = numba_predict_weighted_per_instance_csr_step(
                y_pred.data,
                y_pred.indices,
                y_pred.indptr,
                y_proba.data,
                y_proba.indices,
                y_proba.indptr,
                i,
                self.k,
                0.0,
                self.classifiers_a[c],
                self.classifiers_b[c],
            )
            if y_pred.indices.size != y_pred.data.size:
                y_pred.data = numba_resize(y_pred.data, y_pred.indices.size)
                y_pred.data.fill(1.0)

        else:
            y_pred[i] = predict_using_randomized_weighted_classifier(
                y_proba[i],
                self.classifiers_a,
                self.classifiers_b,
                self.classifiers_proba,
                k=self.k,
            )

    def update_online(self, y_true, y_pred, y_proba, i):
        if self.etu_variant:
            y_true = y_proba

        self.seen_so_far.append(i)
        if (
            len(self.seen_so_far) % self.next_update == self.next_update - 1
            and len(self.seen_so_far) >= self.first_update_n
        ):
            print("  Updating classifier ...")
            rnd_cls = find_classifier_using_fw(
                y_true[self.seen_so_far],
                y_proba[self.seen_so_far],
                self.utility_func,
                max_iters=25,
                k=self.k,
                skip_tn=self.skip_tn,
                alpha_tolerance=0.001,
                alpha_uniform_search_step=0.001,
                verbose=False,
            )
            self.classifiers_a = rnd_cls.a
            self.classifiers_b = rnd_cls.b
            self.classifiers_proba = rnd_cls.p
            self.update_base = floor(self.update_base * self.update_exp)
            self.next_update = len(self.seen_so_far) + self.update_base
            self.updates.append(len(self.seen_so_far))

    def predict_batch(self, y_proba, y_pred, batch):
        y_pred[batch] = predict_using_randomized_weighted_classifier(
            y_proba[batch],
            self.classifiers_a,
            self.classifiers_b,
            self.classifiers_proba,
            k=self.k,
        )

    def get_meta(self):
        return {"updates_steps": self.updates}


class OnlineGreedy(OnlineMethod):
    def __init__(
        self,
        m,
        k,
        binary_utility_func,
        skip_tn=False,
        etu_variant=False,
        # initial_confusion_matrix=(1e-6, 1e-6, 1e-6, 1e-6),
        initial_confusion_matrix=(DEFAULT_REG, DEFAULT_REG, DEFAULT_REG, DEFAULT_REG),
    ):
        super().__init__(m, k)

        self.binary_utility_func = binary_utility_func
        self.skip_tn = skip_tn

        # self.tp = np.zeros(m, dtype=FLOAT_TYPE)
        # self.fp = np.zeros(m, dtype=FLOAT_TYPE)
        # self.fn = np.zeros(m, dtype=FLOAT_TYPE)
        # self.tn = np.zeros(m, dtype=FLOAT_TYPE)
        tp = np.full(m, initial_confusion_matrix[0], dtype=FLOAT_TYPE)
        fp = np.full(m, initial_confusion_matrix[1], dtype=FLOAT_TYPE)
        fn = np.full(m, initial_confusion_matrix[2], dtype=FLOAT_TYPE)
        tn = np.full(m, initial_confusion_matrix[3], dtype=FLOAT_TYPE)
        self.C = ConfusionMatrix(tp, fp, fn, tn)
        self.n = sum(initial_confusion_matrix)

        self.etu_variant = etu_variant
        print("Skip tn", self.skip_tn)

    def predict_online(self, y_proba, y_pred, i):
        if isinstance(y_proba, csr_matrix):
            _bc_with_0approx_step_csr(
                y_proba,
                y_pred,
                i,
                self.C.tp,
                self.C.fp,
                self.C.fn,
                self.C.tn,
                self.k,
                self.binary_utility_func,
                greedy=True,
                skip_tn=self.skip_tn,
                only_pred=True,
            )
        else:
            _bc_with_0approx_step_dense(
                y_proba,
                y_pred,
                i,
                self.C.tp,
                self.C.fp,
                self.C.fn,
                self.C.tn,
                self.k,
                self.binary_utility_func,
                greedy=True,
                skip_tn=self.skip_tn,
                only_pred=True,
            )
        # print(y_pred[i])
        # exit(1)

    def update_online(self, y_true, y_pred, y_proba, i):
        self.n += 1
        if self.etu_variant:
            y_true = y_proba

        _update_unnormalized_confusion_matrix(
            self.C,
            y_true[i],
            y_pred[i],
            skip_tn=self.skip_tn,
        )
        # self.tp += y_true[i] * y_pred[i]
        # self.fp += (1 - y_true[i]) * y_pred[i]
        # self.fn += y_true[i] * (1 - y_pred[i])
        # self.tn += (1 - y_true[i]) * (1 - y_pred[i])

    def has_pu_solution(self):
        return False


class OMMA(OnlineMethod):
    def __init__(
        self,
        m,
        k,
        utility_func,
        buffer_size=1000,
        skip_tn=False,
        etu_variant=False,
        lazy_update=True,
        first_update_n=0,
        # initial_confusion_matrix=(1e-6, 1e-6, 1e-6, 1e-6),
        initial_confusion_matrix=(DEFAULT_REG, DEFAULT_REG, DEFAULT_REG, DEFAULT_REG),
    ):
        super().__init__(m, k)
        self.utility_func = utility_func

        self.classifier_a = np.ones(m, dtype=FLOAT_TYPE)
        self.classifier_b = np.full(m, -0.5, dtype=FLOAT_TYPE)
        self.skip_tn = skip_tn
        self.etu_variant = etu_variant
        self.lazy_update = lazy_update
        self.first_update_n = first_update_n

        # tp = np.zeros(m, dtype=FLOAT_TYPE)
        # fp = np.zeros(m, dtype=FLOAT_TYPE)
        # fn = np.zeros(m, dtype=FLOAT_TYPE)
        # tn = np.zeros(m, dtype=FLOAT_TYPE)
        tp = np.full(m, initial_confusion_matrix[0], dtype=FLOAT_TYPE)
        fp = np.full(m, initial_confusion_matrix[1], dtype=FLOAT_TYPE)
        fn = np.full(m, initial_confusion_matrix[2], dtype=FLOAT_TYPE)
        tn = np.full(m, initial_confusion_matrix[3], dtype=FLOAT_TYPE)
        self.C = ConfusionMatrix(tp, fp, fn, tn)
        self.n = sum(initial_confusion_matrix)

        self.buffer_size = buffer_size
        self.buffer_pos = 0
        self.buffer = np.full(self.buffer_size, -1, dtype=DefaultIndDType)
        self.dim_factor = 0.9999
        print("Skip tn", self.skip_tn)
        print("Lazy", self.lazy_update)

    def predict_online(self, y_proba, y_pred, i):
        if isinstance(y_proba, csr_matrix):
            p_start, p_end = y_pred.indptr[i], y_pred.indptr[i + 1]

            t_start, t_end = y_proba.indptr[i], y_proba.indptr[i + 1]
            t_data = y_proba.data[t_start:t_end]
            t_indices = y_proba.indices[t_start:t_end]

            if self.lazy_update and self.n >= self.first_update_n:
                utility, Gtp, Gfp, Gfn, Gtn = _metric_func_with_gradient_autograd(
                    self.utility_func,
                    self.C.tp[t_indices] / self.n,
                    self.C.fp[t_indices] / self.n,
                    self.C.fn[t_indices] / self.n,
                    self.C.tn[t_indices] / self.n,
                )

                # utility, Gtp, Gfp, Gfn, Gtn = _metric_func_with_gradient_torch(
                #     self.utility_func,
                #     self.C.tp[t_indices] / self.n,
                #     self.C.tp[t_indices] / self.n,
                #     self.C.tp[t_indices] / self.n,
                #     self.C.tp[t_indices] / self.n,
                # )

                # Gtp = Gtp.detach().numpy()
                # Gfp = Gfp.detach().numpy()
                # Gfn = Gfn.detach().numpy()
                # Gtn = Gtn.detach().numpy()

                # new_classifier_a = Gtp - Gfp - Gfn + Gtn
                # new_classifier_b = Gfp - Gtn

                # new_not_nan = ~np.isnan(new_classifier_a)
                # self.classifier_a[t_indices][new_not_nan] = new_classifier_a[new_not_nan]
                # self.classifier_b[t_indices][new_not_nan] = new_classifier_b[new_not_nan]

                self.classifier_a[t_indices] = Gtp - Gfp - Gfn + Gtn
                self.classifier_b[t_indices] = Gfp - Gtn
            (
                y_pred.data,
                y_pred.indices,
                y_pred.indptr,
            ) = numba_predict_weighted_per_instance_csr_step(
                y_pred.data,
                y_pred.indices,
                y_pred.indptr,
                y_proba.data,
                y_proba.indices,
                y_proba.indptr,
                i,
                self.k,
                0.0,
                self.classifier_a,
                self.classifier_b,
            )
            if y_pred.indices.size != y_pred.data.size:
                y_pred.data = numba_resize(y_pred.data, y_pred.indices.size)
                y_pred.data.fill(1.0)

            # gains = t_data * self.classifier_a[t_indices] + self.classifier_b[t_indices]
            # #gains[gains == 0] = np.random.rand(gains[gains == 0].size) #* 1e-6
            # #print(t_data, "Gains", gains)

            # # Update select labels with the best gain and update prediction
            # if self.k > 0:
            #     if gains.size > self.k:
            #         top_k = np.argpartition(-gains, self.k)[:self.k]
            #         y_pred.indices[p_start:p_end] = sorted(t_indices[top_k])
            #     else:
            #         t_indices = np.resize(t_indices, self.k)
            #         t_indices[gains.size :] = 0
            #         y_pred.indices[p_start:p_end] = sorted(t_indices)
            # else:
            #     pred_indices = t_indices[gains >= 0.0]
            #     new_p_end = p_start + pred_indices.size
            #     all_end = y_pred.indptr[-1]
            #     new_all_end = all_end + new_p_end - p_end

            #     #print(pred_indices.shape, p_start, p_end, new_p_end, all_end, new_all_end, y_pred.indices.size, new_p_end - new_all_end, p_end - all_end)
            #     if len(y_pred.indices) < new_all_end:
            #         y_pred.indices = numba_resize(y_pred.indices, len(y_pred.indices) * 2)
            #     if len(y_pred.data) < new_all_end:
            #         y_pred.data = numba_resize(y_pred.data, len(y_pred.data) * 2)
            #         y_pred.data.fill(1.0)
            #         #print(y_pred.indices.size)

            #     y_pred.indices[new_p_end:new_all_end] = y_pred.indices[p_end:all_end]
            #     y_pred.indices[p_start:new_p_end] = pred_indices
            #     y_pred.indptr[i + 1 :] += new_p_end - p_end

        else:
            gains = y_proba[i] * self.classifier_a + self.classifier_b
            if self.k > 0:
                top_k = np.argpartition(-gains, self.k)[: self.k]
                y_pred[i, top_k] = 1.0
            else:
                y_pred[i, gains >= 0.0] = 1.0

    def update_online(self, y_true, y_pred, y_proba, i):
        if self.etu_variant:
            y_true = y_proba

        _update_unnormalized_confusion_matrix(
            self.C,
            y_true[i],
            y_pred[i],
            skip_tn=self.skip_tn,
        )
        self.n += 1

        if not self.lazy_update and self.n >= self.first_update_n:
            utility, Gtp, Gfp, Gfn, Gtn = _metric_func_with_gradient_autograd(
                self.utility_func,
                self.tp / self.n,
                self.fp / self.n,
                self.fn / self.n,
                self.tn / self.n,
            )

            # utility, Gtp, Gfp, Gfn, Gtn = _metric_func_with_gradient_torch(
            #     self.utility_func,
            #     self.C.tp / self.n,
            #     self.C.fp / self.n,
            #     self.C.fn / self.n,
            #     self.C.tn / self.n,
            # )

            # Gtp = Gtp.detach().numpy()
            # Gfp = Gfp.detach().numpy()
            # Gfn = Gfn.detach().numpy()
            # Gtn = Gtn.detach().numpy()

            new_classifier_a = Gtp - Gfp - Gfn + Gtn
            new_classifier_b = Gfp - Gtn

            new_not_nan = ~np.isnan(new_classifier_a)
            self.classifier_a[new_not_nan] = new_classifier_a[new_not_nan]
            self.classifier_b[new_not_nan] = new_classifier_b[new_not_nan]

        # print("Pred pos", y_pred[i].sum())
        # print("Confusion matrix", self.tp, self.fp, self.fn, self.tn)
        # print("Utility", utility, Gtp, Gfp, Gfn, Gtn)
        # print("Classifier", self.classifier_a, self.classifier_b)
        # input()

        # if self.buffer_size > 0 and self.buffer[self.buffer_pos % self.buffer_size] >= 0:
        #     j = self.buffer[self.buffer_pos]
        #     self.tp -= y_pred[j] * y_true[j]
        #     self.fp -= y_pred[j] * (1 - y_true[j])
        #     self.fn -= (1 - y_pred[j]) * y_true[j]
        #     self.tn -= (1 - y_pred[j]) * (1 - y_true[j])
        #     self.n -= 1

        # if self.buffer_size > 0:
        #     self.buffer[self.buffer_pos] = i
        #     self.buffer_pos = (self.buffer_pos + 1) % self.buffer_size

    def get_meta(self):
        return {
            # "a": self.classifier_a,
            # "b": self.classifier_b,
            "thresholds": -self.classifier_b
            / self.classifier_a,
        }


class OnlineFMeasureOptimization(OnlineMethod):
    def __init__(
        self,
        m,
        k,
        etu_variant=False,
        micro=False,
        initial_a=DEFAULT_REG,
        initial_b=DEFAULT_REG,
    ):
        super().__init__(m, k)
        # self.a = np.zeros(m, dtype=FLOAT_TYPE)
        # self.b = np.full(m, 1e-6, dtype=FLOAT_TYPE)
        self.a = np.full(m, initial_a, dtype=FLOAT_TYPE)
        self.b = np.full(m, initial_b, dtype=FLOAT_TYPE)
        self.etu_variant = etu_variant
        self.micro = micro

    def predict_online(self, y_proba, y_pred, i):
        if self.micro:
            th = np.full(self.m, self.a.sum() / self.b.sum())
        else:
            th = self.a / self.b

        if isinstance(y_proba, csr_matrix):
            p_start, p_end = y_pred.indptr[i], y_pred.indptr[i + 1]

            t_start, t_end = y_proba.indptr[i], y_proba.indptr[i + 1]
            t_data = y_proba.data[t_start:t_end]
            t_indices = y_proba.indices[t_start:t_end]

            # pred_indices = t_indices[t_data >= th[t_indices]]
            # new_p_end = p_start + pred_indices.size
            # all_end = y_pred.indptr[-1]
            # new_all_end = all_end + new_p_end - p_end

            # #print(pred_indices.shape, p_start, p_end, new_p_end, all_end, new_all_end, y_pred.indices.size, new_p_end - new_all_end, p_end - all_end)
            # if len(y_pred.indices) < new_all_end:
            #     y_pred.indices = numba_resize(y_pred.indices, len(y_pred.indices) * 2)
            # if len(y_pred.data) < new_all_end:
            #     y_pred.data = numba_resize(y_pred.data, len(y_pred.data) * 2)
            #     y_pred.data.fill(1.0)
            #     #print(y_pred.indices.size)

            # y_pred.indices[new_p_end:new_all_end] = y_pred.indices[p_end:all_end]
            # y_pred.indices[p_start:new_p_end] = pred_indices
            # y_pred.indptr[i + 1 :] += new_p_end - p_end

            y_pred.data, y_pred.indices, y_pred.indptr = numba_set_gains_csr(
                y_pred.data,
                y_pred.indices,
                y_pred.indptr,
                t_data,
                t_indices,
                i,
                0,
                th[t_indices],
            )

            if y_pred.indices.size != y_pred.data.size:
                y_pred.data = numba_resize(y_pred.data, y_pred.indices.size, 1.0)
        else:
            y_pred[i, y_proba[i] >= th] = 1.0

    def update_online(self, y_true, y_pred, y_proba, i):
        if self.etu_variant:
            y_true = y_proba

        if isinstance(y_proba, csr_matrix):
            p_start, p_end = y_pred.indptr[i], y_pred.indptr[i + 1]
            p_data = y_pred.data[p_start:p_end]
            p_indices = y_pred.indices[p_start:p_end]

            t_start, t_end = y_true.indptr[i], y_true.indptr[i + 1]
            t_data = y_true.data[t_start:t_end]
            t_indices = y_true.indices[t_start:t_end]

            tp_data, tp_indices = numba_csr_vec_mul_vec(
                p_data, p_indices, t_data, t_indices
            )

            self.a[tp_indices] += tp_data
            self.b[p_indices] += p_data
            self.b[t_indices] += t_data

        else:
            self.a += y_true[i] * y_pred[i]
            self.b += y_true[i] + y_pred[i]

    def get_meta(self):
        return {"a": self.a, "b": self.b, "thresholds": self.a / self.b}


class OnlineDefault(OnlineMethod):
    def __init__(self, m, k):
        super().__init__(m, k)

    def predict_online(self, y_proba, y_pred, i):
        if isinstance(y_proba, csr_matrix):
            t_start, t_end = y_proba.indptr[i], y_proba.indptr[i + 1]
            t_data = y_proba.data[t_start:t_end]
            t_indices = y_proba.indices[t_start:t_end]
            gains = t_data

            y_pred.data, y_pred.indices, y_pred.indptr = numba_set_gains_csr(
                y_pred.data,
                y_pred.indices,
                y_pred.indptr,
                gains,
                t_indices,
                i,
                self.k,
                0.5,
            )

            # # Update select labels with the best gain and update prediction
            # if self.k > 0:
            #     if gains.size > self.k:
            #         top_k = np.argpartition(-gains, self.k)[:self.k]
            #         y_pred.indices[p_start:p_end] = sorted(t_indices[top_k])
            #     else:
            #         t_indices = np.resize(t_indices, self.k)
            #         t_indices[gains.size :] = 0
            #         y_pred.indices[p_start:p_end] = sorted(t_indices)
            # else:
            #     pred_indices = t_indices[gains >= 0.5]
            #     new_p_end = p_start + pred_indices.size
            #     all_end = y_pred.indptr[-1]
            #     new_all_end = all_end + new_p_end - p_end

            #     #print(pred_indices.shape, p_start, p_end, new_p_end, all_end, new_all_end, y_pred.indices.size, new_p_end - new_all_end, p_end - all_end)
            #     if len(y_pred.indices) < new_all_end:
            #         y_pred.indices = numba_resize(y_pred.indices, len(y_pred.indices) * 2)
            #     if len(y_pred.data) < new_all_end:
            #         y_pred.data = numba_resize(y_pred.data, len(y_pred.data) * 2)
            #         y_pred.data.fill(1.0)
            #         #print(y_pred.indices.size)

            #     y_pred.indices[new_p_end:new_all_end] = y_pred.indices[p_end:all_end]
            #     y_pred.indices[p_start:new_p_end] = pred_indices
            #     y_pred.indptr[i + 1 :] += new_p_end - p_end

        else:
            gains = y_proba[i]
            if self.k > 0:
                top_k = np.argpartition(-gains, self.k)[: self.k]
                y_pred[i, top_k] = 1.0
            else:
                y_pred[i, gains >= 0.5] = 1.0

    def update_online(self, y_true, y_pred, y_proba, i):
        pass


def init_y_pred(y_true, k):
    n, m = y_true.shape
    if isinstance(y_true, csr_matrix):
        per_row = k if k > 0 else 10
        data = np.ones(n * per_row, dtype=FLOAT_TYPE)
        indices = np.ones(n * per_row, dtype=DefaultIndDType)
        indptr = np.arange(n + 1, dtype=DefaultIndDType) * per_row
        y_pred = csr_matrix((data, indices, indptr), shape=(n, m))
    else:
        y_pred = np.zeros((n, m), dtype=FLOAT_TYPE)

    return y_pred


def online_experiment(
    online_method_class,
    online_method_args,
    utility_func,
    y_proba,
    y_true,
    k,
    seed=None,
    epochs=1,
    log_current_perf_every=100,
    evaluate_every=1000000,
    batch_size=1,
    shuffle_order=True,
    **kwargs,
):
    print(
        f"  Starting online experiment with {y_proba.shape}, shuffled={shuffle_order}, seed={seed}, epochs={epochs}, batch_size={batch_size} ..."
    )
    y_preds = []
    np.random.seed(seed)

    n, m = y_proba.shape

    if n > 100000:
        log_current_perf_every = 1000

    rng = np.random.default_rng(seed)
    order = np.arange(n)

    meta = {
        "time": time(),
        "iters": epochs,
        "pred_utility_history": [],
        "solution_utility_history": [],
    }
    eval_time = 0

    if shuffle_order:
        rng.shuffle(order)

    online_method_args.update({"m": m, "k": k})
    method = online_method_class(**online_method_args)
    skip_tn = online_method_args.get("skip_tn", False)

    # tp = np.zeros(m, dtype=FLOAT_TYPE)
    # fp = np.zeros(m, dtype=FLOAT_TYPE)
    # fn = np.zeros(m, dtype=FLOAT_TYPE)
    # tn = np.zeros(m, dtype=FLOAT_TYPE)
    tp = np.zeros(m, dtype=np.float64)
    fp = np.zeros(m, dtype=np.float64)
    fn = np.zeros(m, dtype=np.float64)
    tn = np.zeros(m, dtype=np.float64)
    C = ConfusionMatrix(tp, fp, fn, tn)

    print(f"  Starting online experiment")
    for e in range(epochs):
        print(f"    Startin epoch {e + 1} / {epochs} ...")

        y_pred = init_y_pred(y_true, k)

        for i, s in enumerate(order):
            method.predict_online(y_proba, y_pred, s)
            method.update_online(y_true, y_pred, y_proba, s)

            if s == 1:
                print(y_pred[s])

            eval_time = time()
            _update_unnormalized_confusion_matrix(
                C, y_true[s], y_pred[s], skip_tn=skip_tn
            )
            c = i + 1

            if i % log_current_perf_every == log_current_perf_every - 1 or i == n - 1:
                utility = utility_func(C.tp / c, C.fp / c, C.fn / c, C.tn / c)
                meta["pred_utility_history"].append((i, float(utility)))
                print(f"      {i + 1} / {n} utility so far: {utility:.5f}")
                # print(f"C: {C.tp / c}, {C.fp / c}, {C.fn / c}, {C.tn / c}")

            # if method.has_pu_solution() and (
            #     i % evaluate_every == evaluate_every - 1 or i == n - 1
            # ):
            #     _y_pred = init_y_pred(y_true, k)
            #     method.predict_batch(y_proba, _y_pred, order)
            #     _tp, _fp, _fn, _tn = calculate_confusion_matrix(
            #         y_true, _y_pred, normalize=True
            #     )
            #     utility = utility_func(_tp, _fp, _fn, _tn)
            #     meta["solution_utility_history"].append((i, float(utility)))
            #     print(f"      {i + 1} / {n} solution utility so far: {utility:.5f}")

            eval_time = time() - eval_time
            meta["time"] += eval_time

    print(y_pred.shape)

    meta.update(method.get_meta())
    y_preds.append(y_pred)

    meta["time"] = time() - meta["time"]
    return y_preds, meta


def online_default_micro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineDefault,
        {},
        micro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_macro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineDefault,
        {},
        macro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_macro_recall(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")
    return online_experiment(
        OnlineDefault,
        {},
        macro_recall_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_macro_precision(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")
    return online_experiment(
        OnlineDefault,
        {},
        macro_precision_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_macro_hmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineDefault,
        {},
        macro_hmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_macro_gmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineDefault,
        {},
        macro_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_multiclass_gmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OnlineDefault,
        {},
        multiclass_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_multiclass_hmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OnlineDefault,
        {},
        multiclass_hmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_multiclass_qmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OnlineDefault,
        {},
        multiclass_qmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_default_macro_min_tp_tn(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineDefault,
        {},
        macro_min_tp_tn_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


# Online Greedy


def online_greedy_macro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineGreedy,
        {
            "binary_utility_func": binary_f1_score_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_greedy_micro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    raise ValueError("this is not working")

    return online_experiment(
        OnlineGreedy,
        {
            "binary_utility_func": micro_f1_score_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        micro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_greedy_macro_recall(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")

    return online_experiment(
        OnlineGreedy,
        {
            "binary_utility_func": binary_recall_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_recall_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_greedy_macro_precision(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")

    return online_experiment(
        OnlineGreedy,
        {
            "binary_utility_func": binary_precision_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_precision_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_greedy_macro_min_tp_tn(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineGreedy,
        {
            "binary_utility_func": binary_min_tp_tn_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_min_tp_tn_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_greedy_macro_hmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineGreedy,
        {
            "binary_utility_func": binary_hmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_hmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_greedy_macro_gmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineGreedy,
        {
            "binary_utility_func": binary_gmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


# Online Frank Wolfe


def online_frank_wolfe_macro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": macro_f1_score_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_micro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": micro_f1_score_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        micro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_macro_recall(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")

    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": macro_recall_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_recall_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_macro_precision(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")

    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": macro_precision_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_precision_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_macro_min_tp_tn(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": macro_min_tp_tn_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_min_tp_tn_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_macro_hmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": macro_hmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_hmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_macro_gmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": macro_gmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_multiclass_hmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": multiclass_hmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        multiclass_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_multiclass_gmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": log_multiclass_gmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        multiclass_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_frank_wolfe_multiclass_qmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OnlineFrankWolfe,
        {
            "utility_func": multiclass_qmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        multiclass_qmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


# Online Thresholds


def online_thresholds_macro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineThresholds,
        {"binary_utility_func": binary_f1_score_on_conf_matrix},
        macro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_thresholds_micro_f1(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineThresholds,
        {"binary_utility_func": binary_f1_score_on_conf_matrix, "micro": True},
        micro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_thresholds_macro_recall(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")
    return online_experiment(
        OnlineThresholds,
        {"binary_utility_func": binary_recall_on_conf_matrix},
        macro_recall_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_thresholds_macro_precision(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")
    return online_experiment(
        OnlineThresholds,
        {"binary_utility_func": binary_precision_on_conf_matrix},
        macro_precision_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def online_thresholds_macro_min_tp_tn(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OnlineThresholds,
        {"binary_utility_func": binary_min_tp_tn_on_conf_matrix},
        macro_min_tp_tn_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


# My Online


def omma_macro_f1(y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs):
    return online_experiment(
        OMMA,
        {
            "utility_func": macro_f1_score_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_micro_f1(y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs):
    return online_experiment(
        OMMA,
        {
            "utility_func": micro_f1_score_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        micro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_macro_recall(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")
    return online_experiment(
        OMMA,
        {
            "utility_func": macro_recall_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_recall_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_macro_precision(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k == 0:
        raise ValueError("k must be > 0")
    return online_experiment(
        OMMA,
        {
            "utility_func": macro_precision_on_conf_matrix,
            "skip_tn": True,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_precision_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_macro_min_tp_tn(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OMMA,
        {
            "utility_func": macro_min_tp_tn_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_min_tp_tn_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_macro_hmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OMMA,
        {
            "utility_func": macro_hmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_hmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_macro_gmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    return online_experiment(
        OMMA,
        {
            "utility_func": macro_gmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        macro_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_multiclass_gmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OMMA,
        {
            "utility_func": log_multiclass_gmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        multiclass_gmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_multiclass_hmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OMMA,
        {
            "utility_func": multiclass_hmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        multiclass_hmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def omma_multiclass_qmean(
    y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs
):
    if k != 1:
        raise ValueError("k must be 1")
    return online_experiment(
        OMMA,
        {
            "utility_func": multiclass_qmean_on_conf_matrix,
            "etu_variant": kwargs.get("etu_variant", False),
        },
        multiclass_qmean_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def ofo_macro(y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs):
    if k != 0:
        raise ValueError("k must be equal to 0")

    return online_experiment(
        OnlineFMeasureOptimization,
        {"etu_variant": kwargs.get("etu_variant", False)},
        macro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )


def ofo_micro(y_proba, k: int = 5, seed: int = 0, y_true=None, epochs=1, **kwargs):
    if k != 0:
        raise ValueError("k must be equal to 0")

    return online_experiment(
        OnlineFMeasureOptimization,
        {"etu_variant": kwargs.get("etu_variant", False), "micro": True},
        micro_f1_score_on_conf_matrix,
        y_proba,
        y_true,
        k=k,
        seed=seed,
        epochs=epochs,
        **kwargs,
    )
