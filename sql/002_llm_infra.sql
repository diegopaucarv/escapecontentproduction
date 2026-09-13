-- =====================================================================
-- Pipeline Unificado de Producción de Contenidos — ESCAPE / Ergalia
-- 002_llm_infra.sql — infraestructura LLM (proveedor Together AI)
--
-- Dos tablas:
--   1. api_keys          — claves de API por proveedor (la clave NUNCA
--                          vive en código ni en git; se inserta vía
--                          seed/env, ver src/db/seed_llm.py).
--   2. session_settings  — settings de sesión: modelos pequeño/grande
--                          + parámetros, referenciando UNA api_key.
--                          Solo una fila activa a la vez (singleton).
-- =====================================================================

-- ---------------------------------------------------------------------
-- 1. Claves de API
-- ---------------------------------------------------------------------

CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    provider VARCHAR(50) NOT NULL,          -- 'together' | 'openai' | ...
    key_name VARCHAR(100) NOT NULL,         -- etiqueta legible, ej. 'together-main'
    api_key TEXT NOT NULL,                  -- la clave en sí (secreto)
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (provider, key_name)
);

CREATE TRIGGER trg_api_keys_updated_at
    BEFORE UPDATE ON api_keys
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------------------
-- 2. Settings de sesión (modelos pequeño/grande)
-- ---------------------------------------------------------------------

CREATE TABLE session_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    api_key_id UUID NOT NULL REFERENCES api_keys(id),
    small_model VARCHAR(150) NOT NULL,
    large_model VARCHAR(150) NOT NULL,
    temperature_small DECIMAL(3,2) NOT NULL DEFAULT 0.7,
    temperature_large DECIMAL(3,2) NOT NULL DEFAULT 0.7,
    max_tokens_small INT NOT NULL DEFAULT 2048,
    max_tokens_large INT NOT NULL DEFAULT 4096,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Singleton: a lo sumo UNA fila activa a la vez.
CREATE UNIQUE INDEX uq_session_settings_active
    ON session_settings (is_active) WHERE is_active;

CREATE TRIGGER trg_session_settings_updated_at
    BEFORE UPDATE ON session_settings
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();
