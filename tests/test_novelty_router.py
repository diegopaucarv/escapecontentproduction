from src.agents.novelty_router import NoveltyInputs, score_novelty


def _base_inputs(**overrides) -> NoveltyInputs:
    defaults = dict(
        brand_objective="ESCAPE_SOCIAL",
        content_bucket="difusion_cientifica",
        artifact_type="video_corto",
        channel="tiktok",
        insight_core="algo nunca antes cubierto",
        bucket_ever_used_by_brand=True,
        artifact_type_ever_used_by_brand=True,
        channel_ever_used_by_brand=True,
    )
    defaults.update(overrides)
    return NoveltyInputs(**defaults)


def test_all_familiar_still_scores_angulo_nuevo_because_no_duplicate_was_found():
    # Si llegamos a score_novelty es porque el Paso 1 (búsqueda de
    # duplicado) ya falló en encontrar algo comparable -> el ángulo
    # siempre suma, aunque bucket/formato/canal sean conocidos.
    inputs = _base_inputs()
    assert score_novelty(inputs) == 3  # solo angulo_nuevo


def test_new_bucket_and_channel_adds_up():
    inputs = _base_inputs(bucket_ever_used_by_brand=False, channel_ever_used_by_brand=False)
    # angulo_nuevo(3) + bucket_nuevo(3) + canal_nuevo(2) = 8
    assert score_novelty(inputs) == 8


def test_custom_weights_are_respected():
    inputs = _base_inputs(bucket_ever_used_by_brand=False, weights={
        "bucket_nuevo": 10, "formato_nuevo": 0, "canal_nuevo": 0, "angulo_nuevo": 0,
    })
    assert score_novelty(inputs) == 10
