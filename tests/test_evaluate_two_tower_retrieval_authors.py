import argparse
import asyncio
import importlib.util
import sys
from datetime import date
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def retrieval_eval_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "ops/evaluate_two_tower_retrieval_authors.py"
    spec = importlib.util.spec_from_file_location("evaluate_two_tower_retrieval_authors", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_two_tower_retrieval_authors"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeTower:
    def __init__(self, output=None, passthrough=False):
        self.output = output
        self.passthrough = passthrough
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        if self.passthrough:
            return args[0]
        return self.output


def _model(module, *, run_id="model-a", use_author=False, user_tower=None, post_tower=None):
    return module.ModelBundle(
        run_id=run_id,
        train_dir=Path(f"/tmp/{run_id}"),
        training_config={
            "max_history_len": 3,
            "post_embedding_dim": 2,
            "use_author_embedding_table": use_author,
        },
        manifest={"inputs": {}},
        user_tower=user_tower or FakeTower(torch.tensor([[1.0, 0.0]])),
        post_tower=post_tower or FakeTower(passthrough=True),
        author_idx_by_did={"did:a": 7} if use_author else None,
    )


def test_default_dates_and_es_host_normalization(retrieval_eval_module):
    assert retrieval_eval_module.default_date_range(date(2026, 7, 5)) == ("2026-07-03", "2026-07-05")
    assert retrieval_eval_module.normalize_es_host("localhost:9200") == "https://localhost:9200"
    assert retrieval_eval_module.normalize_es_host("https://example.com/") == "https://example.com"
    assert retrieval_eval_module.is_local_https("https://localhost:9200")


def test_post_url_helpers_parse_at_uri(retrieval_eval_module):
    at_uri = "at://did:plc:author/app.bsky.feed.post/abc123"

    assert retrieval_eval_module.get_post_id(at_uri) == "abc123"
    assert retrieval_eval_module.at_uri_to_url(at_uri) == "https://bsky.app/profile/did:plc:author/post/abc123"
    assert retrieval_eval_module.safe_at_uri_to_url("not-an-at-uri") is None

    with pytest.raises(ValueError, match="Could not find post ID"):
        retrieval_eval_module.get_post_id("not-an-at-uri")


def test_no_author_tower_signatures_and_scores(retrieval_eval_module):
    user_tower = FakeTower(torch.tensor([[1.0, 0.0]]))
    post_tower = FakeTower(passthrough=True)
    model = _model(retrieval_eval_module, user_tower=user_tower, post_tower=post_tower)
    history_posts = [
        retrieval_eval_module.HistoryPost(rank=1, at_uri="h1", author_did="did:a", embedding=[0.5, 0.5]),
    ]

    user_embedding = retrieval_eval_module.compute_user_embedding_for_model(model, history_posts, torch.device("cpu"))
    assert torch.equal(user_embedding, torch.tensor([1.0, 0.0]))
    assert len(user_tower.calls[0]) == 2

    candidates = [
        retrieval_eval_module.CandidatePost("p1", "did:a", "2026-07-03T00:00:00", [0.1, 0.9]),
        retrieval_eval_module.CandidatePost("p2", "did:b", "2026-07-03T00:00:00", [0.8, 0.2]),
    ]
    scores = retrieval_eval_module.score_candidate_batch_for_model(model, user_embedding, candidates, torch.device("cpu"))

    assert len(post_tower.calls[0]) == 1
    torch.testing.assert_close(scores, torch.tensor([0.1, 0.8]))


def test_author_aware_towers_map_known_and_unknown_authors(retrieval_eval_module):
    user_tower = FakeTower(torch.tensor([[1.0, 0.0]]))
    post_tower = FakeTower(passthrough=True)
    model = _model(
        retrieval_eval_module,
        use_author=True,
        user_tower=user_tower,
        post_tower=post_tower,
    )
    history_posts = [
        retrieval_eval_module.HistoryPost(rank=1, at_uri="h1", author_did="did:a", embedding=[0.5, 0.5]),
        retrieval_eval_module.HistoryPost(rank=2, at_uri="h2", author_did="did:unknown", embedding=[0.2, 0.8]),
    ]

    user_embedding = retrieval_eval_module.compute_user_embedding_for_model(model, history_posts, torch.device("cpu"))
    assert torch.equal(user_embedding, torch.tensor([1.0, 0.0]))
    assert len(user_tower.calls[0]) == 3
    history_author_indices = user_tower.calls[0][2]
    assert history_author_indices.tolist() == [[7, retrieval_eval_module.AUTHOR_UNK_IDX, 0]]

    candidates = [
        retrieval_eval_module.CandidatePost("p1", "did:a", "2026-07-03T00:00:00", [0.3, 0.7]),
        retrieval_eval_module.CandidatePost("p2", "did:missing", "2026-07-03T00:00:00", [0.9, 0.1]),
    ]
    scores = retrieval_eval_module.score_candidate_batch_for_model(model, user_embedding, candidates, torch.device("cpu"))

    assert len(post_tower.calls[0]) == 2
    target_author_indices = post_tower.calls[0][1]
    assert target_author_indices.tolist() == [7, retrieval_eval_module.AUTHOR_UNK_IDX]
    torch.testing.assert_close(scores, torch.tensor([0.3, 0.9]))


def test_history_posts_preserve_es_like_order_and_debug_fields(retrieval_eval_module):
    liked_uris = ["at://post/2", "at://post/1", "at://post/3"]
    posts_by_uri = {
        "at://post/1": {
            "at_uri": "at://post/1",
            "author_did": "did:a",
            "embeddings": {"all_MiniLM_L12_v2": [0.1, 0.2]},
        },
        "at://post/2": {
            "at_uri": "at://post/2",
            "author_did": "did:b",
            "embeddings": {"all_MiniLM_L12_v2": [0.3, 0.4]},
        },
    }

    history_posts = retrieval_eval_module.build_history_posts_from_es(
        liked_uris,
        posts_by_uri,
        embedding_model="all_MiniLM_L12_v2",
        embed_dim=2,
    )
    rows = retrieval_eval_module.history_posts_json(history_posts, {"did:a": "a.test", "did:b": "b.test"})

    assert [post.at_uri for post in history_posts] == liked_uris
    assert rows[0]["at_uri"] == "at://post/2"
    assert rows[0]["author_did"] == "did:b"
    assert rows[0]["embedding_present"] is True
    assert rows[2]["embedding_present"] is False


def test_topk_accumulator_and_author_counts_match_exact_scores(retrieval_eval_module):
    accumulator = retrieval_eval_module.TopKAccumulator(2)
    first_batch = [
        retrieval_eval_module.CandidatePost("p1", "did:a", "2026-07-03T00:00:00", [0.0, 0.0]),
        retrieval_eval_module.CandidatePost("p2", "did:b", "2026-07-03T00:00:00", [0.0, 0.0]),
        retrieval_eval_module.CandidatePost("p3", "did:a", "2026-07-03T00:00:00", [0.0, 0.0]),
    ]
    second_batch = [
        retrieval_eval_module.CandidatePost("p4", "did:c", "2026-07-03T00:00:00", [0.0, 0.0]),
    ]

    accumulator.add(torch.tensor([0.5, 0.1, 0.8]), first_batch)
    accumulator.add(torch.tensor([0.7]), second_batch)
    ranked = accumulator.ranked("model-a")
    counts = retrieval_eval_module.build_author_counts(ranked, {"did:a": "a.test", "did:c": "c.test"})

    assert [post.at_uri for post in ranked] == ["p3", "p4"]
    assert [post.score for post in ranked] == [0.800000011920929, 0.699999988079071]
    assert counts == [
        {"author_did": "did:a", "handle": "a.test", "count": 1, "best_rank": 1},
        {"author_did": "did:c", "handle": "c.test", "count": 1, "best_rank": 2},
    ]


def test_output_dir_and_json_artifacts_are_written_by_default(retrieval_eval_module, tmp_path):
    output_dir = retrieval_eval_module.create_output_dir(
        None,
        output_root=tmp_path,
        timestamp="20260706_120000",
        short_uuid="abc12345",
    )
    model_a = _model(retrieval_eval_module, run_id="model-a")
    model_b = _model(retrieval_eval_module, run_id="model-b")
    args = argparse.Namespace(
        train_run_ids=["model-a", "model-b"],
        train_artifacts_dir=Path("/models"),
        user_did="did:plc:user",
        start_date="2026-07-03",
        end_date="2026-07-05",
        top_k=50,
        es_host="https://localhost:9200",
        history_limit=50,
        candidate_batch_size=8192,
    )
    top_posts_by_model = {
        "model-a": [
            retrieval_eval_module.TopPost(
                "model-a",
                1,
                0.9,
                "at://did:a/app.bsky.feed.post/p1",
                "did:a",
                "2026-07-03T00:00:00",
                "post one",
            ),
        ],
        "model-b": [
            retrieval_eval_module.TopPost(
                "model-b",
                1,
                0.8,
                "at://did:b/app.bsky.feed.post/p2",
                "did:b",
                "2026-07-03T00:00:00",
                "post two",
            ),
        ],
    }
    author_counts_by_model = {
        run_id: retrieval_eval_module.build_author_counts(posts, {})
        for run_id, posts in top_posts_by_model.items()
    }
    summary = retrieval_eval_module.build_summary(
        args=args,
        output_dir=output_dir,
        models=[model_a, model_b],
        author_counts_by_model=author_counts_by_model,
        top_posts_by_model=top_posts_by_model,
        n_candidate_posts_scanned=2,
        n_history_likes=1,
        n_embeddable_history_posts=1,
        gcs_bucket="bucket",
        embedding_model="all_MiniLM_L12_v2",
    )
    history_rows = [{"at_uri": "h1", "author_did": "did:h"}]
    top_k_rows = retrieval_eval_module.top_k_posts_json(top_posts_by_model, {})

    retrieval_eval_module.write_output_artifacts(
        output_dir=output_dir,
        summary=summary,
        history_rows=history_rows,
        top_k_rows=top_k_rows,
    )

    assert output_dir == tmp_path / "20260706_120000_abc12345"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "history_posts.json").exists()
    assert (output_dir / "top_k_posts.json").exists()
    assert set(summary["models"].keys()) == {"model-a", "model-b"}
    assert top_k_rows[0]["at_uri"] == "at://did:a/app.bsky.feed.post/p1"
    assert top_k_rows[0]["url"] == "https://bsky.app/profile/did:a/post/p1"
    assert top_k_rows[0]["author_did"] == "did:a"
    assert top_k_rows[0]["content"] == "post one"


def test_resolve_author_handles_falls_back_to_did(retrieval_eval_module):
    async def fake_fetch_profile(_client, did):
        if did == "did:a":
            return "a.test"
        if did == "did:b":
            raise RuntimeError("lookup failed")
        return None

    handles = asyncio.run(
        retrieval_eval_module.resolve_author_handles(
            ["did:c", "did:a", "did:b"],
            fetch_profile=fake_fetch_profile,
        )
    )

    assert handles == {
        "did:a": "a.test",
        "did:b": "did:b",
        "did:c": "did:c",
    }


def test_missing_artifacts_fail_clearly(retrieval_eval_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="missing required artifact"):
        retrieval_eval_module.resolve_model_paths(tmp_path, "missing-run")

    with pytest.raises(FileNotFoundError, match="No author_idx_.*parquet"):
        retrieval_eval_module.resolve_author_idx_path(tmp_path)
