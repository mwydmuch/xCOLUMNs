class Metric(object):
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return f"{self.name}:"
    


class MetricOnConfusionMatrix(Metric):
    def __init__(self, name):
        super().__init__(name)

    def calculate_on_confusion_matrix(self, tp, fp, fn, tn):
        raise NotImplementedError()
    
    def torch_calculate_on_confusion_matrix(self, tp, fp, fn, tn):
        return self.calculate_on_confusion_matrix(tp, fp, fn, tn)

    def calculate(self, y_true, y_pred):
        tp, fp, fn, tn = self.calculate_confusion_matrix(y_true, y_pred)
        return self.value_on_confusion_matrix(tp, fp, fn, tn)


    def __str__(self):
        return f"{super().__str__()} tp: {self.tp} fp: {self.fp} fn: {self.fn} tn: {self.tn}"