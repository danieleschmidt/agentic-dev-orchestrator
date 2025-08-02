# Fish completion for ado (Agentic Development Orchestrator)

# Define completion commands
complete -c ado -f

# Commands
complete -c ado -n '__fish_use_subcommand' -a 'init' -d 'Initialize ADO in current directory'
complete -c ado -n '__fish_use_subcommand' -a 'run' -d 'Execute autonomous backlog processing'
complete -c ado -n '__fish_use_subcommand' -a 'status' -d 'Show current backlog status'
complete -c ado -n '__fish_use_subcommand' -a 'discover' -d 'Discover new backlog items from code'
complete -c ado -n '__fish_use_subcommand' -a 'help' -d 'Show help information'

# Options
complete -c ado -n '__fish_use_subcommand' -l help -s h -d 'Show help information'

# Same completions for the full name
complete -c agentic-dev-orchestrator -f
complete -c agentic-dev-orchestrator -n '__fish_use_subcommand' -a 'init' -d 'Initialize ADO in current directory'
complete -c agentic-dev-orchestrator -n '__fish_use_subcommand' -a 'run' -d 'Execute autonomous backlog processing'
complete -c agentic-dev-orchestrator -n '__fish_use_subcommand' -a 'status' -d 'Show current backlog status'
complete -c agentic-dev-orchestrator -n '__fish_use_subcommand' -a 'discover' -d 'Discover new backlog items from code'
complete -c agentic-dev-orchestrator -n '__fish_use_subcommand' -a 'help' -d 'Show help information'
complete -c agentic-dev-orchestrator -n '__fish_use_subcommand' -l help -s h -d 'Show help information'