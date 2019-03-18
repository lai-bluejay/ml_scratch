def cal_a_b(odds, p, base):
    """
    给定odds和分base， 翻倍分段p，
    返回A，B
    例如：好坏比2：1，希望是950。1：1，900
  
    Arguments:
        odds {[type]} -- [description]
        p {[type]} -- [description]
        base {[type]} -- [description]
    
    Returns:
        [a] -- [预设分数]
        [b] -- [factor]
    """
    b = p / np.log(2)
    a = base + b * np.log(odds)
    return a, b


class ScoreCard():
    """[summary]
    
    Returns:
        [type] -- [description]
    """

    def __init__(self, base=100, o=1, p=20, round_=1, threshold=None):
        self.b = base
        self.o = o
        self.p = p
        self.threshold = threshold
        self.round_ = round_
        self.offset = self.b - self.p * (np.log(self.o) / np.log(2))
        self.factor = self.p / np.log(2)

    def predict(self, pt):
        """用于预测某一条预处理过的特征向量得分的方法
        Parameters:
        pt: - 违约概率
        Returns:
        float: - 预测出来的分数
        bool: - 预测的分数超过阈值则返回True,否则False
        """
        proba = [[pt, 1 - pt]]
        p_f, p_t = proba[0]
        odds = p_t / p_f
        score = round(self.factor * np.log(odds) + self.offset, self.round_)
        if self.threshold:
            return self.threshold if score > self.threshold else score
        else:
            return score

        def map_od_rate(self, score):
            odds = np.exp((score - self.offset) / self.factor)
            pt = odds / (1 + odds)
            return pt
