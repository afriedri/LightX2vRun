# MiniMax-H3 FL2VA prompt selection shared by the host and container entries.

resolve_minimax_h3_prompt() {
    local run_root="$1"
    local preset="$2"
    local prompt_file="$3"
    local duration_seconds="$4"
    local template_path=
    local resolved_prompt=
    local resolved_source=

    if [[ -n "${PROMPT:-}" ]]; then
        resolved_prompt="${PROMPT}"
        resolved_source=environment:PROMPT
    elif [[ -n "${prompt_file}" ]]; then
        [[ -f "${prompt_file}" ]] || {
            echo "ERROR: MiniMax-H3 prompt file not found: ${prompt_file}" >&2
            return 2
        }
        resolved_prompt="$(<"${prompt_file}")"
        resolved_source="file:${prompt_file}"
    else
        case "${preset}" in
            bird-stable | bird-dynamic | bird-audio)
                template_path="${run_root}/prompts/minimax_h3/${preset}.txt"
                ;;
            *)
                echo "ERROR: unknown MiniMax-H3 prompt preset '${preset}'; expected bird-stable, bird-dynamic, or bird-audio" >&2
                return 2
                ;;
        esac
        [[ -f "${template_path}" ]] || {
            echo "ERROR: MiniMax-H3 prompt preset file not found: ${template_path}" >&2
            return 2
        }
        resolved_prompt="$(<"${template_path}")"
        resolved_source="preset:${preset}"
    fi

    resolved_prompt="${resolved_prompt//\{\{DURATION_SECONDS\}\}/${duration_seconds}}"
    [[ -n "${resolved_prompt}" ]] || {
        echo "ERROR: MiniMax-H3 prompt must not be empty" >&2
        return 2
    }
    [[ "${resolved_prompt}" != *'{{DURATION_SECONDS}}'* ]] || {
        echo "ERROR: unresolved MiniMax-H3 prompt duration placeholder" >&2
        return 2
    }

    MINIMAX_H3_PROMPT="${resolved_prompt}"
    MINIMAX_H3_PROMPT_SOURCE="${resolved_source}"
}
