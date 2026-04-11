cd ~/arca

cat > context.txt << 'EOF'
# ================================================
# ARCA — Autonomous Reinforcement Cyber Agent
# Full Codebase Context — Generated on $(date)
# ================================================

EOF

# Add all important files with clear headers
echo -e "\n\n# ==================== ROOT FILES ====================" >> context.txt
for file in pyproject.toml README.md setup.py .gitignore; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file\n# ================================================" >> context.txt
        cat "$file" >> context.txt
    fi
done

echo -e "\n\n# ==================== ARCA PACKAGE FILES ====================" >> context.txt

# Main package files
for file in arca/__init__.py arca/__version__.py arca/cli.py; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file\n# ================================================" >> context.txt
        cat "$file" >> context.txt
    fi
done

# Core
echo -e "\n\n# ------------------- Core -------------------" >> context.txt
for file in arca/core/*.py; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file\n# ================================================" >> context.txt
        cat "$file" >> context.txt
    fi
done

# Sim
echo -e "\n\n# ------------------- Sim -------------------" >> context.txt
for file in arca/sim/*.py; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file\n# ================================================" >> context.txt
        cat "$file" >> context.txt
    fi
done

# C++ Extension
echo -e "\n\n# ------------------- C++ Extension -------------------" >> context.txt
for file in arca/cpp_ext/*.cpp arca/cpp_ext/*.py; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file\n# ================================================" >> context.txt
        cat "$file" >> context.txt
    fi
done

# Agents, Viz, API, Utils
for dir in agents viz api utils; do
    echo -e "\n\n# ------------------- $dir -------------------" >> context.txt
    for file in arca/$dir/*.py; do
        if [ -f "$file" ]; then
            echo -e "\n\n# File: $file\n# ================================================" >> context.txt
            cat "$file" >> context.txt
        fi
    done
done

# Examples, Tests
echo -e "\n\n# ==================== EXAMPLES & TESTS ====================" >> context.txt
for file in examples/*.py tests/*.py; do
    if [ -f "$file" ]; then
        echo -e "\n\n# File: $file\n# ================================================" >> context.txt
        cat "$file" >> context.txt
    fi
done

echo -e "\n\n# ================================================" >> context.txt
echo "# End of ARCA codebase context" >> context.txt
echo "# Total files included: $(find arca -name "*.py" -o -name "*.cpp" -o -name "pyproject.toml" -o -name "README.md" | wc -l)" >> context.txt

echo "✅ context.txt created successfully with all files!"
echo "File location: $(pwd)/context.txt"
echo "Size: $(wc -c < context.txt) bytes"
ls -lh context.txt