from src.freshness.clustering import (
    cluster_items,
    normalize_title,
    title_tokens,
    token_similarity,
)


def test_normalize_title_strips_case_and_punctuation():
    assert normalize_title("Reliance's Q1 Profit UP 12%!") == "reliances q1 profit up 12"
    assert normalize_title("  multiple   spaces  ") == "multiple spaces"
    assert normalize_title("") == ""
    assert normalize_title(None) == ""


def test_title_tokens_drops_common_stopwords_but_never_empties_a_real_title():
    toks = title_tokens("Reliance shares slip after SEBI warning")
    assert "reliance" in toks and "sebi" in toks and "warning" in toks
    assert "after" not in toks          # stopword removed
    # a title made entirely of stopwords keeps its tokens rather than becoming empty
    assert title_tokens("of the and to") == frozenset({"of", "the", "and", "to"})


def test_token_similarity_is_length_robust_overlap():
    # overlap coefficient: |A cap B| / min(|A|,|B|) -- robust to one headline being a longer rewrite
    assert token_similarity(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0
    assert token_similarity(frozenset({"a"}), frozenset({"b"})) == 0.0
    assert token_similarity(frozenset(), frozenset()) == 0.0   # two empties do not merge
    # a shared core across differing lengths still scores high (this is why not plain Jaccard)
    a = frozenset({"reliance", "sebi", "warning"})
    b = frozenset({"reliance", "sebi", "warning", "stock", "falls", "today", "again"})
    assert token_similarity(a, b) == 1.0        # a fully inside b
    assert 0.4 < token_similarity(frozenset({"reliance", "sebi", "warning", "shares", "slip"}), b) < 0.7


def test_identical_titles_cluster_together():
    titles = [
        "Reliance shares slip after SEBI warning",
        "Reliance shares slip after SEBI warning",
    ]
    clusters = cluster_items(titles)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_near_duplicate_rewrites_collapse_to_one_cluster():
    # WHY: the same story reworded by N outlets must collapse, or one event floods the feed.
    titles = [
        "Reliance shares slip after SEBI warning",
        "SEBI warning sends Reliance shares slipping",       # reworded, same story
        "Reliance stock falls following SEBI warning today",  # reworded again
        "Tata Motors quarterly profit jumps 20 percent",     # unrelated -> own cluster
    ]
    clusters = cluster_items(titles, threshold=0.5)
    # the three Reliance rewrites collapse; Tata stands alone
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 3]


def test_unrelated_stories_stay_separate():
    titles = [
        "Infosys wins large cloud deal in Europe",
        "HDFC Bank raises deposit rates",
        "ONGC discovers new gas field offshore",
    ]
    clusters = cluster_items(titles, threshold=0.5)
    assert len(clusters) == 3


def test_cluster_items_works_on_objects_via_key():
    class Item:
        def __init__(self, title):
            self.title = title
    items = [Item("Reliance shares slip after SEBI warning"),
             Item("SEBI warning sends Reliance shares slipping"),
             Item("Adani ports posts record cargo volume")]
    clusters = cluster_items(items, key=lambda x: x.title, threshold=0.5)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]


def test_non_latin_titles_are_tokenized_not_dropped():
    # WHY (product): Hindi/regional press breaks stories first; those titles must still cluster.
    hindi_a = "रिलायंस के शेयरों में भारी गिरावट"
    hindi_b = "टाटा मोटर्स का तिमाही मुनाफा बढ़ा"
    clusters = cluster_items([hindi_a, hindi_a, hindi_b], threshold=0.5)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 2]     # the two identical Hindi items cluster; the different one is separate


def test_empty_titles_do_not_merge_into_one_bucket():
    # Two content-free titles are not provably the same story; keep them apart.
    clusters = cluster_items(["", "", "Real headline here about markets"], threshold=0.5)
    assert len(clusters) == 3
