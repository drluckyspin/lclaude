#!/bin/bash

# -----------------------------------------------------------------------------------------------------------
# Script Name: log.bash
# Version: 1.9
#
# Description: A collection of logging utility functions for bash scripts that
#              provide colored and formatted console output.
# -----------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------------------------------------
: "${VERBOSE:=false}"

# -----------------------------------------------------------------------------------------------------------
# ANSI color codes
# -----------------------------------------------------------------------------------------------------------
RED='\033[91m'
GREEN='\033[92m'
YELLOW='\033[93m'
BLUE='\033[94m'
WHITE='\033[97m'
PURPLE='\033[95m'
RESET='\033[0m'
DIM='\033[2m'

# -----------------------------------------------------------------------------------------------------------
# Function: get_terminal_width
# Description: Get current terminal width (max 120 chars)
# -----------------------------------------------------------------------------------------------------------
get_terminal_width() {
    local width
    width="$(tput cols 2>/dev/null || echo 80)"
    if [[ -z "$width" || "$width" -le 0 ]]; then
        width=80
    elif [ "$width" -gt 120 ]; then
        width=120
    fi
    echo "$width"
}

# -----------------------------------------------------------------------------------------------------------
# Basic logging functions
# -----------------------------------------------------------------------------------------------------------
log() {
    echo -e "${WHITE} $1${RESET}"
}

log_dim() {
    echo -e "${DIM}${WHITE} $1${RESET}"
}

log_info() {
    echo -e "${BLUE} $1${RESET}"
}

log_info_dim() {
    echo -e "${DIM}${BLUE} $1${RESET}"
}

log_success() {
    echo -e " ${GREEN}✔${RESET} ${DIM}${GREEN} $1${RESET}"
}

log_error() {
    echo -e " ${RED}🅇${RESET}  ${DIM}${RED}$1${RESET}"
}

log_warning() {
    echo -e " ${YELLOW}▲${RESET}  ${DIM}${YELLOW}$1${RESET}"
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_separator
# Description: Print a separator line across terminal width
# -----------------------------------------------------------------------------------------------------------
log_separator() {
    local terminal_width
    terminal_width=$(get_terminal_width)
    printf "=-%.0s" $(seq 1 $((terminal_width / 2)))
    echo "="
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_target
# Description: Open a make target — one separator, blank line, then title.
# -----------------------------------------------------------------------------------------------------------
log_target() {
    log_separator
    echo ""
    log_info "$1"
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_section
# Description: Start a section within a target — blank line, then title.
# -----------------------------------------------------------------------------------------------------------
log_section() {
    echo ""
    log_info "$1"
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_indent
# Description: Indent (2 spaces) and call any log function.
# -----------------------------------------------------------------------------------------------------------
log_indent() {
    local log_func=$1
    shift
    printf "  "
    $log_func "$@"
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_pipe_dim
# Description: Stream stdin with 2-space indent and dim styling.
# -----------------------------------------------------------------------------------------------------------
log_pipe_dim() {
    while IFS= read -r line || [ -n "$line" ]; do
        printf "  ${DIM}${WHITE}%s${RESET}\n" "$line"
    done
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_run_dim
# Description: Run a command; pipe combined stdout/stderr through log_pipe_dim.
# -----------------------------------------------------------------------------------------------------------
log_run_dim() {
    "$@" 2>&1 | log_pipe_dim
    return "${PIPESTATUS[0]}"
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_pipe_info
# Description: Stream stdin with 2-space indent and info styling.
# -----------------------------------------------------------------------------------------------------------
log_pipe_info() {
    while IFS= read -r line || [ -n "$line" ]; do
        printf "  ${DIM}%s${RESET}\n" "$line"
    done
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_run_info
# Description: Run a command; pipe combined stdout/stderr through log_pipe_info.
# -----------------------------------------------------------------------------------------------------------
log_run_info() {
    "$@" 2>&1 | log_pipe_info
    return "${PIPESTATUS[0]}"
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_centered
# Description: Center a message in the terminal.
# -----------------------------------------------------------------------------------------------------------
log_centered() {
    local terminal_width
    local message="$1"
    terminal_width=$(get_terminal_width)

    local padding=$(((terminal_width - ${#message}) / 2))
    local pad_str
    pad_str=$(printf '%*s' "$padding" '')
    echo -e "${pad_str}${message}"
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_verbose
# Description: Log message only if VERBOSE environment variable is true.
# -----------------------------------------------------------------------------------------------------------
log_verbose() {
    if [[ "${VERBOSE:-false}" == "true" ]]; then
        log_info_dim "$*"
    fi
}

# -----------------------------------------------------------------------------------------------------------
# Function: log_banner
# Description: Display the lclaude project banner.
# -----------------------------------------------------------------------------------------------------------
log_banner() {
    echo -e "
    ${PURPLE}██╗      ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗
    ${PURPLE}██║     ██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝
    ${PURPLE}██║     ██║     ██║     ███████║██║   ██║██║  ██║█████╗
    ${PURPLE}██║     ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝
    ${PURPLE}███████╗╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗
    ${PURPLE}╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
    ${RESET}"
}

# -----------------------------------------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------------------------------------
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    log_separator
    log_banner
    log "This is a normal message."
    log_dim "This is a dim message."
    log_info "This is an info message."
    log_info_dim "This is a dim info message."
    log_success "This is a success message."
    log_warning "This is a warning message."
    log_error "This is an error message."
    log_indent log_success "This is an indented message."
    log_centered "This is a centered message"
    log_separator
fi

