"""Flujo agéntico de proyectos (Fase 5, diseño §20.3).

El LLM es un COMPILADOR: escribe datos (el snapshot JSON), nunca edita
archivos. Las reglas deterministas viven en el código, no en los prompts.
Este paquete conecta la entidad Project (0007) con el pipeline de
producción existente (brief → artefacto → manifiesto → orquestador).
"""
