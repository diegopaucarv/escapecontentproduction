"""infraestructura de IA modular: llm_models, prompt_templates, prompt_artifacts, embedding_settings

Registro de modelos (llm_models), specs agnósticas de prompts (prompt_templates),
artefactos compilados e inmutables (prompt_artifacts) y settings de embeddings
(embedding_settings). Además añade fallback_model y llm_retries a session_settings.

Revision ID: 0004_ai_modular_infra
Revises: 0003_llm_infra
Create Date: 2026-09-12
"""

from alembic import op

revision = "0004_ai_modular_infra"
down_revision = "0003_llm_infra"
branch_labels = None
depends_on = None

UPGRADE_SQL = r"""
-- ---------------------------------------------------------------------
-- 1. Registro de modelos LLM (CRUD-editable)
-- ---------------------------------------------------------------------

CREATE TABLE llm_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(150) NOT NULL UNIQUE,
    provider VARCHAR(50) NOT NULL,            -- 'together' | 'voyage' | 'openai' | ...
    model_size VARCHAR(20) NOT NULL,          -- 'small' | 'large' | 'embedding'
    context_window INT NOT NULL DEFAULT 0,
    max_output_tokens INT NOT NULL DEFAULT 0,
    temperature_default DECIMAL(3,2) NOT NULL DEFAULT 0.7,
    strengths JSONB NOT NULL DEFAULT '[]'::jsonb,
    weaknesses JSONB NOT NULL DEFAULT '[]'::jsonb,
    prompt_style TEXT NOT NULL DEFAULT '',
    syntax_profile JSONB NOT NULL DEFAULT '{}'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_llm_models_updated_at
    BEFORE UPDATE ON llm_models
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 2. Specs agnósticas de prompts (CRUD-editable, NO archivos YAML)
-- ---------------------------------------------------------------------

CREATE TABLE prompt_templates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_key VARCHAR(100) NOT NULL UNIQUE,    -- 'alignment_reinforcement' | 'critic_checklist' | ...
    version VARCHAR(20) NOT NULL DEFAULT '1.0',
    intent TEXT NOT NULL,
    rules JSONB NOT NULL DEFAULT '[]'::jsonb,
    input_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    few_shot JSONB NOT NULL DEFAULT '[]'::jsonb,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TRIGGER trg_prompt_templates_updated_at
    BEFORE UPDATE ON prompt_templates
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 3. Artefactos compilados — INMUTABLES (nunca UPDATE; recompilar =
--    INSERT nueva versión + desactivar la anterior)
-- ---------------------------------------------------------------------

CREATE TABLE prompt_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    llm_model_id UUID NOT NULL REFERENCES llm_models(id),
    task_key VARCHAR(100) NOT NULL,
    spec_version VARCHAR(20) NOT NULL,
    artifact_version INT NOT NULL,
    prompt_text TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,        -- sha256 hex
    compiled_by VARCHAR(100) NOT NULL DEFAULT 'compiler',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (llm_model_id, task_key, artifact_version)
);

-- Solo UNA versión activa por (modelo, tarea):
CREATE UNIQUE INDEX uq_prompt_artifacts_active
    ON prompt_artifacts (llm_model_id, task_key) WHERE is_active;

-- ---------------------------------------------------------------------
-- 4. Settings de embeddings (singleton)
-- ---------------------------------------------------------------------

CREATE TABLE embedding_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID NOT NULL REFERENCES api_keys(id),
    llm_model_id UUID NOT NULL REFERENCES llm_models(id),
    dimension INT NOT NULL DEFAULT 1024,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX uq_embedding_settings_active
    ON embedding_settings (is_active) WHERE is_active;

CREATE TRIGGER trg_embedding_settings_updated_at
    BEFORE UPDATE ON embedding_settings
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 5. session_settings: fallback + reintentos
-- ---------------------------------------------------------------------

ALTER TABLE session_settings ADD COLUMN fallback_model VARCHAR(150) NULL;
ALTER TABLE session_settings ADD COLUMN llm_retries INT NOT NULL DEFAULT 3;
"""

DOWNGRADE_SQL = r"""
ALTER TABLE session_settings DROP COLUMN IF EXISTS llm_retries;
ALTER TABLE session_settings DROP COLUMN IF EXISTS fallback_model;

DROP TABLE IF EXISTS embedding_settings CASCADE;
DROP TABLE IF EXISTS prompt_artifacts CASCADE;
DROP TABLE IF EXISTS prompt_templates CASCADE;
DROP TABLE IF EXISTS llm_models CASCADE;
"""


def upgrade() -> None:
    op.execute(UPGRADE_SQL)


def downgrade() -> None:
    op.execute(DOWNGRADE_SQL)
