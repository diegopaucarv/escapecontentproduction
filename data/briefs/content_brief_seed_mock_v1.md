---
tipo: brief
origen: seed-mock-v1
usar_para: desarrollo local / pruebas del endpoint GET /briefs/{id}
---

# Content Brief — seed-mock-v1

> Brief de ejemplo generado por `python -m src.db.seed`. El estado
> vigente vive en la tabla `content_briefs`; este archivo es el espejo
> legible en Markdown (ver `data/templates/prefill_content_brief.md`).

## Identificación
- owner: (sin asignar)
- status: revision

## Origen y contexto
- resumen: Explicar en 60 segundos por qué la vacunación de refuerzo sigue siendo relevante en 2026, con datos del último estudio de inmunidad poblacional.
- insight_core: La gente cree que el refuerzo ya no es necesario; los datos de inmunidad poblacional muestran lo contrario.
- pitch_15s: ¿Crees que ya no necesitas el refuerzo? Los datos dicen otra cosa.
- prior_attempts: Un post estático en 2024 con poca interacción.
- risks: Riesgo de polarización en comentarios sobre vacunas.
- suggested_product_type: video_corto
- org_priorities_contrast: Prioridad de la marca: desmentir mitos de salud pública.

## Jerarquía canónica
- brand_objective: ESCAPE_SOCIAL
- segment_client: S1
- audience_tier: tier_1
- need: Información confiable sobre salud pública.

## Bucket y oferta
- content_bucket: difusion_cientifica
- entry_offer: guía_descargable

## Artefacto y canal
- artifact_type: video_corto
- channel: tiktok
- cta: Descarga la guía completa
- funnel_stage: top

## Gobernanza y evidencia
- evidence_source: Estudio de inmunidad poblacional 2026 (fuente primaria).
- risk_level: medio
- validation_required: ["factual", "legal"]
- repurpose_plan: [{"artifact_type": "post", "channel": "instagram"}, {"artifact_type": "hilo", "channel": "x"}]

## Métricas
- metric_primary: retention_24h
- metric_secondary: conversion_30d
