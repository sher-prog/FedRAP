import os
import random
import pandas as pd

from utils.data import SampleGenerator


def _read_inter(path):
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = df.rename(columns={"user_id:token": "user", "item_id:token": "item"})
    df = df[df["item"] != ""]
    return df[["user", "item"]].reset_index(drop=True)


class Paper2Data:
    def __init__(self, data_dir, eps):
        clean = _read_inter(os.path.join(data_dir, "ml1m.train.inter"))
        self.user_ids = sorted(clean["user"].unique(), key=int)
        self.item_ids = sorted(clean["item"].unique(), key=int)
        self.num_users = len(self.user_ids)
        self.num_items = len(self.item_ids)
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: i for i, it in enumerate(self.item_ids)}
        if eps == "nodp":
            raw = clean
        else:
            raw = _read_inter(os.path.join(data_dir, "ml1m.train_e{}.inter".format(eps)))
        self.train = self._toIndices(raw)
        self.train["rating"] = 1.0
        self.train["timestamp"] = 0
        self.train_items = self._groupItems(self.train)
        val = self._toIndices(_read_inter(os.path.join(data_dir, "ml1m.val.inter")))
        self.val_items = self._groupItems(val)

    def _toIndices(self, df):
        df = df[df["user"].isin(self.user_index.keys()) & df["item"].isin(self.item_index.keys())]
        return pd.DataFrame({
            "userId": df["user"].map(self.user_index).astype(int).to_numpy(),
            "itemId": df["item"].map(self.item_index).astype(int).to_numpy(),
        })

    def _groupItems(self, df):
        items = {u: set() for u in range(self.num_users)}
        for u, i in zip(df["userId"], df["itemId"]):
            items[int(u)].add(int(i))
        return items


class Paper2SampleGenerator(SampleGenerator):
    def __init__(self, data):
        self.data = data
        self.item_pool = set(range(data.num_items))
        self.train_ratings = data.train[["userId", "itemId", "rating"]]

    def store_all_train_data(self, num_negatives):
        num_items = self.data.num_items
        users, items, ratings = [], [], []
        for u in range(self.data.num_users):
            liked = self.data.train_items[u]
            single_user, user_item, user_rating = [], [], []
            for i in sorted(liked):
                single_user.append(u)
                user_item.append(i)
                user_rating.append(1.0)
                for neg in self._sampleNegatives(liked, num_negatives, num_items):
                    single_user.append(u)
                    user_item.append(neg)
                    user_rating.append(0.0)
            users.append(single_user)
            items.append(user_item)
            ratings.append(user_rating)
        assert len(users) == len(items) == len(ratings) == self.data.num_users
        return [users, items, ratings]

    @staticmethod
    def _sampleNegatives(liked, num_negatives, num_items):
        if num_items - len(liked) <= num_negatives * 4:
            candidates = [i for i in range(num_items) if i not in liked]
            return random.sample(candidates, min(num_negatives, len(candidates)))
        picked = set()
        while len(picked) < num_negatives:
            j = random.randrange(num_items)
            if j not in liked:
                picked.add(j)
        return list(picked)
