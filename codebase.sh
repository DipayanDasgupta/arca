#!/bin/bash
# ================================================
# ARCA Full Codebase Generator - Reliable v3
# Usage: bash generate_full_codebase.sh
# ================================================

OUTPUT_FILE="ARCA_Full_Codebase_$(date +%Y%m%d_%H%M).md"

cat > "$OUTPUT_FILE" << EOF
# ARCA Complete Codebase Snapshot

**Generated:** $(date)
**Project:** ARCA v3.2 — GNN + CleanRL-PPO + LocalLLM + Curriculum + Vector Memory
**Total files:** (processing...)

---

## 📋 Table of Contents
*(Will be updated at the end)*

---

EOF

echo "📦 Starting full codebase export..."

file_count=0
toc=""

while IFS= read -r -d '' filepath; do
    rel_path="${filepath#./}"
    
    # Skip junk
    if [[ "$rel_path" == *__pycache__* ]] || 
       [[ "$rel_path" == *arca_outputs* ]] || 
       [[ "$rel_path" == *dist* ]] || 
       [[ "$rel_path" == *build* ]] || 
       [[ "$rel_path" == *egg-info* ]] || 
       [[ "$rel_path" == *test_visuals* ]] || 
       [[ "$rel_path" == *.so ]] || 
       [[ "$rel_path" == *vector_cache* ]]; then
        continue
    fi

    # Only include relevant file types
    if [[ "$rel_path" != *.py ]] && 
       [[ "$rel_path" != *.cpp ]] && 
       [[ "$rel_path" != *.sh ]] && 
       [[ "$rel_path" != *.toml ]] && 
       [[ "$rel_path" != *README* ]] && 
       [[ "$rel_path" != *MANIFEST.in ]] && 
       [[ "$rel_path" != *setup.py ]] && 
       [[ "$rel_path" != *.yaml ]] && 
       [[ "$rel_path" != *.md ]]; then
        continue
    fi

    ((file_count++))

    # Build TOC entry
    toc="${toc}- [$rel_path](#file-$(echo "$rel_path" | tr '/.' '-'))\n"

    echo "✓ Adding: $rel_path"

    cat >> "$OUTPUT_FILE" << EOF

## File: \`$rel_path\`
<a name="file-$(echo "$rel_path" | tr '/.' '-')"></a>

EOF

    # Language detection
    case "$rel_path" in
        *.py)   lang="python" ;;
        *.cpp)  lang="cpp" ;;
        *.sh)   lang="bash" ;;
        *.toml) lang="toml" ;;
        *.yaml|*.yml) lang="yaml" ;;
        *.md)   lang="markdown" ;;
        *)      lang="" ;;
    esac

    echo "\`\`\`$lang" >> "$OUTPUT_FILE"
    cat "$filepath" >> "$OUTPUT_FILE"
    echo "\`\`\`" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo "---" >> "$OUTPUT_FILE"

done < <(find . -type f -print0 | sort -z)

# Finalize TOC and summary
sed -i "s/Total files: (processing...)/Total files: **$file_count**/" "$OUTPUT_FILE"
sed -i "s|\*(Will be updated at the end)\*|$toc|" "$OUTPUT_FILE"

cat >> "$OUTPUT_FILE" << EOF

---

## Summary
- **Files included:** $file_count
- **Excluded:** __pycache__, outputs, dist, build, .git, .so, caches
- **Generated on:** $(date)

Ready for Claude / Cursor / Gemini.
EOF

echo ""
echo "✅ Done! Full codebase saved to:"
echo "   $OUTPUT_FILE"
echo "   Total files: $file_count"
echo ""
echo "Open it with:"
echo "   code \"$OUTPUT_FILE\""