#!/bin/bash
# =============================================================================
# RALPH LOOP: Radiomics.jl - Pure Julia Port of PyRadiomics
# =============================================================================
# This loop orchestrates an autonomous agent to port PyRadiomics to Julia
# with 1:1 unit test parity using PythonCall.jl for verification.
#
# Usage: ./loop.sh [max_iterations]
# Example: ./loop.sh 100
# =============================================================================

set -euo pipefail

# Configuration
MAX_ITERATIONS="${1:-100}"
COOLDOWN_SECONDS=3
AGENT_TIMEOUT_SECONDS=$((30 * 60))  # 30 minutes max per iteration

# Paths (relative to repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PRD_FILE="$SCRIPT_DIR/prd.json"
PROMPT_FILE="$SCRIPT_DIR/prompt.md"
PROGRESS_FILE="$SCRIPT_DIR/progress.md"
GUARDRAILS_FILE="$SCRIPT_DIR/guardrails.md"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_required_files() {
    local missing=0
    for file in "$PRD_FILE" "$PROMPT_FILE" "$PROGRESS_FILE" "$GUARDRAILS_FILE"; do
        if [[ ! -f "$file" ]]; then
            log_error "Missing required file: $file"
            missing=1
        fi
    done
    return $missing
}

count_open_stories() {
    grep -c '"status": "open"' "$PRD_FILE" 2>/dev/null || echo "0"
}

count_done_stories() {
    grep -c '"status": "done"' "$PRD_FILE" 2>/dev/null || echo "0"
}

show_spinner() {
    local pid=$1
    local delay=1
    local elapsed=0
    local spinstr='|/-\'

    while ps -p "$pid" > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " [%c] Elapsed: %dm %ds  " "$spinstr" $((elapsed / 60)) $((elapsed % 60))
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        elapsed=$((elapsed + 1))
        printf "\r"

        # Check timeout
        if [[ $elapsed -ge $AGENT_TIMEOUT_SECONDS ]]; then
            log_warning "Agent timeout reached ($AGENT_TIMEOUT_SECONDS seconds)"
            kill "$pid" 2>/dev/null || true
            return 1
        fi
    done
    printf "                                    \r"
    return 0
}

# =============================================================================
# Main Loop
# =============================================================================

main() {
    log_info "Starting Radiomics.jl Ralph Loop"
    log_info "Max iterations: $MAX_ITERATIONS"
    log_info "Working directory: $REPO_ROOT"

    # Verify required files exist
    if ! check_required_files; then
        log_error "Missing required files. Exiting."
        exit 1
    fi

    cd "$REPO_ROOT"

    local iteration=0

    while [[ $iteration -lt $MAX_ITERATIONS ]]; do
        iteration=$((iteration + 1))

        # Count stories
        local open_count=$(count_open_stories)
        local done_count=$(count_done_stories)

        echo ""
        log_info "=========================================="
        log_info "ITERATION $iteration / $MAX_ITERATIONS"
        log_info "Stories: $done_count done, $open_count open"
        log_info "=========================================="

        # Check if all stories are complete
        if [[ "$open_count" -eq 0 ]]; then
            log_success "All stories complete! Ralph loop finished successfully."
            exit 0
        fi

        # Log iteration start to progress file
        echo "" >> "$PROGRESS_FILE"
        echo "### Iteration $iteration - $(date '+%Y-%m-%d %H:%M:%S')" >> "$PROGRESS_FILE"
        echo "" >> "$PROGRESS_FILE"
        echo "**Agent started** (Open: $open_count, Done: $done_count)" >> "$PROGRESS_FILE"
        echo "" >> "$PROGRESS_FILE"

        # Run the agent
        log_info "Launching Claude agent..."

        local output_file=$(mktemp)

        # Run claude with the prompt, capturing output
        claude --print --dangerously-skip-permissions < "$PROMPT_FILE" > "$output_file" 2>&1 &
        local agent_pid=$!

        # Show spinner while waiting
        if ! show_spinner "$agent_pid"; then
            log_warning "Agent was terminated due to timeout"
        fi

        # Wait for agent to finish and get exit code
        wait "$agent_pid" 2>/dev/null || true

        # Check output for completion markers
        if grep -q "<promise>COMPLETE</promise>" "$output_file" 2>/dev/null; then
            log_success "Agent signaled story completion"
        fi

        if grep -q "<promise>BLOCKED</promise>" "$output_file" 2>/dev/null; then
            log_error "Agent signaled BLOCKED - human intervention required"
            echo "" >> "$PROGRESS_FILE"
            echo "**BLOCKED** - Agent requires human intervention" >> "$PROGRESS_FILE"
            cat "$output_file" >> "$PROGRESS_FILE"
            rm -f "$output_file"
            exit 1
        fi

        if grep -q "<promise>ALL_COMPLETE</promise>" "$output_file" 2>/dev/null; then
            log_success "Agent signaled ALL stories complete!"
            rm -f "$output_file"
            exit 0
        fi

        rm -f "$output_file"

        # Cooldown before next iteration
        log_info "Cooling down for $COOLDOWN_SECONDS seconds..."
        sleep $COOLDOWN_SECONDS

    done

    log_warning "Max iterations ($MAX_ITERATIONS) reached without completion"
    exit 1
}

# Run main
main "$@"
