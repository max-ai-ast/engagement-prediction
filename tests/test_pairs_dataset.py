import pandas as pd

from utils.helpers import create_pairs_dataset


def test_create_pairs_dataset_generates_negatives_without_overlap():
    posts_emb_df = pd.DataFrame(
        {
            "post_id": ["p1", "p2", "p3"],
            "post_emb_0": [0.1, 0.2, 0.3],
            "post_emb_1": [1.0, 2.0, 3.0],
        }
    )
    likes_df = pd.DataFrame(
        {
            "did": ["u1", "u1", "u2"],
            "post_id": ["p1", "p2", "p1"],
        }
    )

    dataset = create_pairs_dataset(
        likes_df,
        posts_emb_df,
        join_like="post_id",
        join_post="post_id",
        random_seed=0,
        use_parallel=False,
    )

    assert {"did", "post_id", "post_emb_0", "liked"} <= set(dataset.columns)
    pos = dataset[dataset["liked"] == 1]
    neg = dataset[dataset["liked"] == 0]

    assert len(pos) == 3
    assert len(neg) == 2
    positive_pairs = set(zip(pos["did"], pos["post_id"]))
    negative_pairs = set(zip(neg["did"], neg["post_id"]))
    assert positive_pairs.isdisjoint(negative_pairs)
