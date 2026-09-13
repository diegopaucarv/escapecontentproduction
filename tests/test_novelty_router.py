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
    inputs = _base_inputs(
        bucket_ever_used_by_brand=False, channel_ever_used_by_brand=False
    )
    # angulo_nuevo(3) + bucket_nuevo(3) + canal_nuevo(2) = 8
    assert score_novelty(inputs) == 8


def test_custom_weights_are_respected():
    inputs = _base_inputs(
        bucket_ever_used_by_brand=False,
        weights={
            "bucket_nuevo": 10,
            "formato_nuevo": 0,
            "canal_nuevo": 0,
            "angulo_nuevo": 0,
        },
    )
    assert score_novelty(inputs) == 10


# ---------------------------------------------------------------------
# route_brief_with_refinement — coordinador de la zona gris (0004)
# ---------------------------------------------------------------------


class _FakeSession:
    """Sesión mínima: route_brief no la usa si find_closest_prior_artifact
    está mockeado."""

    def execute(self, stmt):
        class _Result:
            def scalars(self):
                return self

            def first(self):
                return None

        return _Result()


def _match(**overrides):
    from types import SimpleNamespace

    defaults = dict(
        id=1,
        content_summary="pieza previa sobre el mismo tema",
        retention_24h=0.5,
        conversion_30d=0.1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_route_brief_with_refinement_uses_gray_zone(monkeypatch):
    import src.agents.novelty_router as router_mod
    import src.llm.novelty_refinement as refinement_mod

    # Zona gris: similitud 0.75 (entre 0.70 y 0.86).
    monkeypatch.setattr(
        router_mod,
        "find_closest_prior_artifact",
        lambda session, embed_fn, insight, objective: (_match(), 0.75),
    )
    # El LLM dice que el ángulo ya está cubierto.
    monkeypatch.setattr(
        refinement_mod,
        "refine_angle_novelty",
        lambda session, brief_data, prior_artifacts: {
            "status": "ok",
            "angulo_nuevo": False,
            "reasoning": "ya cubierto",
        },
    )

    inputs = _base_inputs()  # todo familiar -> score base 3 (solo ángulo)
    result = router_mod.route_brief_with_refinement(
        _FakeSession(), lambda s: [0.1] * 1024, inputs
    )
    # score 3 - 3 (ángulo cubierto) = 0 -> repetitivo
    assert result.route_decision == "repetitivo"
    assert result.novelty_score == 0


def test_route_brief_with_refinement_passes_prior_artifact(monkeypatch):
    import src.agents.novelty_router as router_mod
    import src.llm.novelty_refinement as refinement_mod

    captured = {}
    monkeypatch.setattr(
        router_mod,
        "find_closest_prior_artifact",
        lambda session, embed_fn, insight, objective: (_match(), 0.75),
    )

    def fake_refine(session, brief_data, prior_artifacts):
        captured["prior"] = prior_artifacts
        return {"status": "ok", "angulo_nuevo": True, "reasoning": "nuevo"}

    monkeypatch.setattr(refinement_mod, "refine_angle_novelty", fake_refine)

    # Bucket nuevo: score = angulo(3) + bucket(3) = 6 -> nueva_solucion.
    inputs = _base_inputs(bucket_ever_used_by_brand=False)
    result = router_mod.route_brief_with_refinement(
        _FakeSession(), lambda s: [0.1] * 1024, inputs
    )
    assert result.route_decision == "nueva_solucion"
    assert result.novelty_score == 6
    assert captured["prior"] == [
        {
            "id": "1",
            "content_summary": "pieza previa sobre el mismo tema",
            "retention_24h": 0.5,
            "conversion_30d": 0.1,
        }
    ]


def test_route_brief_with_refinement_skips_clear_new(monkeypatch):
    """Fuera de la zona gris (similitud < 0.70) el LLM NO se consulta."""
    import src.agents.novelty_router as router_mod
    import src.llm.novelty_refinement as refinement_mod

    called = {"n": 0}
    monkeypatch.setattr(
        router_mod,
        "find_closest_prior_artifact",
        lambda session, embed_fn, insight, objective: (_match(), 0.30),
    )

    def fake_refine(session, brief_data, prior_artifacts):
        called["n"] += 1
        return {"status": "ok", "angulo_nuevo": False, "reasoning": "x"}

    monkeypatch.setattr(refinement_mod, "refine_angle_novelty", fake_refine)

    inputs = _base_inputs()
    result = router_mod.route_brief_with_refinement(
        _FakeSession(), lambda s: [0.1] * 1024, inputs
    )
    assert called["n"] == 0  # nunca se consultó al LLM
    # Todo familiar: score 3 (solo ángulo) < umbral 4 -> repetitivo.
    assert result.route_decision == "repetitivo"
    assert result.novelty_score == 3  # ángulo nuevo por definición


def test_route_brief_with_refinement_degraded_keeps_default(monkeypatch):
    """Si el refinador degrada (LLM no disponible), el enrutamiento no cambia."""
    import src.agents.novelty_router as router_mod
    import src.llm.novelty_refinement as refinement_mod

    monkeypatch.setattr(
        router_mod,
        "find_closest_prior_artifact",
        lambda session, embed_fn, insight, objective: (_match(), 0.75),
    )
    monkeypatch.setattr(
        refinement_mod,
        "refine_angle_novelty",
        lambda session, brief_data, prior_artifacts: {
            "status": "degraded",
            "reason": "llm_unavailable",
            "angulo_nuevo": True,
        },
    )

    inputs = _base_inputs()
    result = router_mod.route_brief_with_refinement(
        _FakeSession(), lambda s: [0.1] * 1024, inputs
    )
    # Degradado -> default determinista: score 3 < umbral 4 -> repetitivo.
    assert result.route_decision == "repetitivo"
    assert result.novelty_score == 3  # default determinista intacto
