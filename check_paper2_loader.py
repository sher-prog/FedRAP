""" Checks utlis/paper2_data.py against Paper 2 - Mufliha's ML-1M files."""

import sys
from utils.paper2_data import Paper2Data, Paper2SampleGenerator

DATA_DIR = sys.argv[1]
failures = 0

def check(name, got, expected):
    global failures
    ok = got == expected
    failures += not ok
    print(("PASS " if ok else "FAIL ") + f"{name}: got {got!r}" + ("" if ok else f", expected {expected!r}"))

EXPECTED = {
    "nodp": {"rows": 340356, "empty_users": []},
    "1":    {"rows": 340675, "empty_users": ["1621", "4295"]},
    "0.1":  {"rows": 339676, "empty_users": ["4365", "5370"]},
}

for eps, exp in EXPECTED.items():
    print(f"\n--- eps = {eps} ---")
    d = Paper2Data(DATA_DIR, eps)

    # Fixed index from the CLEAN train file, identical for every eps
    check("num_users", d.num_users, 6035)
    check("num_items", d.num_items, 3430)
    check("user_ids are original IDs as strings", (type(d.user_ids[0]).__name__, d.user_ids[0]), ("str", "1"))
    check("excluded users absent (4349, 4636, 5850)", any(u in d.user_ids for u in ["4349", "4636", "5850"]), False)

    # Training rows
    check("train rows", len(d.train), exp["rows"])
    check("train columns", list(d.train.columns), ["userId", "itemId", "rating", "timestamp"])
    check("indices in range", (int(d.train.userId.min()), int(d.train.userId.max()) < 6035,
                               int(d.train.itemId.min()), int(d.train.itemId.max()) < 3430), (0, True, 0, True))

    # The user who lost every like to the noise are kept and with an empty set
    check("train_items has every user", len(d.train_items), 6035)
    empty = sorted(d.user_ids[u] for u, s in d.train_items.items() if len(s) == 0)
    check("users with no training likes", empty, exp["empty_users"])

    # No leave-one-out and no 10 like filter: user '1' keeps all clean train likes
    if eps == "nodp":
        check("user '1' train likes", len(d.train_items[d.user_ids.index("1")]), 27)
        check("val_items has every user", len(d.val_items), 6035)

    # Sample generator keeps every user at its own position
    g = Paper2SampleGenerator(d)
    users, items, ratings = g.store_all_train_data(num_negatives=4)
    check("one training list per user", len(users), 6035)
    u0 = d.user_ids.index("1")
    check("user '1' rows = likes x (1 + 4 negatives)", len(users[u0]), len(d.train_items[u0]) * 5)
    check("list position matches user index", set(users[u0]), {u0})
    for u in empty:
        check(f"user {u} has an empty list", len(users[d.user_ids.index(u)]), 0)

print("\nALL PASS" if failures == 0 else f"\n{failures} FAIL(S)")

    


