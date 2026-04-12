#!/bin/bash
# generate_context.sh — Creates a comprehensive context.txt for Claude / LLM

cd ~/arca

CONTEXT_FILE="context.txt"

cat > "$CONTEXT_FILE" << 'HEADER'
# ================================================
# ARCA — Autonomous Reinforcement Cyber Agent
# Full Codebase Context for LLM Improvement
# Generated on: $(date)
# Version: 0.1.1 (PyPI)
# GitHub: https://github.com/DipayanDasgupta/arca
# ================================================

HEADER

echo -e "\n\n# ==================== ROOT FILES ====================" >> "$CONTEXT_FILE"

# Root files
for file in pyproject.toml README.md setup.py .gitignore; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file" >> "$CONTEXT_FILE"
        echo "# ================================================" >> "$CONTEXT_FILE"
        cat "$file" >> "$CONTEXT_FILE"
        echo -e "\n" >> "$CONTEXT_FILE"
    fi
done

echo -e "\n\n# ==================== ARCA PACKAGE ====================" >> "$CONTEXT_FILE"

# Main package files
for file in arca/__init__.py arca/__version__.py arca/cli.py; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file" >> "$CONTEXT_FILE"
        echo "# ================================================" >> "$CONTEXT_FILE"
        cat "$file" >> "$CONTEXT_FILE"
        echo -e "\n" >> "$CONTEXT_FILE"
    fi
done

# Core, sim, agents, viz, api, utils, cpp_ext
for dir in core sim agents viz api utils cpp_ext; do
    echo -e "\n\n# ------------------- $dir -------------------" >> "$CONTEXT_FILE"
    for file in arca/$dir/*.py arca/$dir/*.cpp; do
        if [ -f "$file" ]; then
            echo -e "\n\n# File: $file" >> "$CONTEXT_FILE"
            echo "# ================================================" >> "$CONTEXT_FILE"
            cat "$file" >> "$CONTEXT_FILE"
            echo -e "\n" >> "$CONTEXT_FILE"
        fi
    done
done

echo -e "\n\n# ==================== EXAMPLES & TESTS ====================" >> "$CONTEXT_FILE"

for file in examples/*.py tests/*.py; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file" >> "$CONTEXT_FILE"
        echo "# ================================================" >> "$CONTEXT_FILE"
        cat "$file" >> "$CONTEXT_FILE"
        echo -e "\n" >> "$CONTEXT_FILE"
    fi
done

echo -e "\n\n# ================================================" >> "$CONTEXT_FILE"
echo "# End of ARCA codebase context" >> "$CONTEXT_FILE"
echo "# Total files included: $(find arca -name "*.py" -o -name "*.cpp" -o -name "pyproject.toml" -o -name "README.md" -o -name "setup.py" | wc -l)" >> "$CONTEXT_FILE"

echo "✅ context.txt generated successfully!"
echo "Location: $(pwd)/context.txt"
ls -lh context.txt