# Fonts for Synthetic Plates

The license plate generator attempts to load fonts in this priority order:

1. `FE-Schrift.ttf` — The correct typeface for Mercosul plates (German FE-Schrift / Fälschungserschwerende Schrift)
2. `FESchrift.ttf` — Alternate naming
3. `DejaVuSansMono-Bold.ttf` — Fallback monospace font (place here)
4. System DejaVu Sans Mono Bold (`/usr/share/fonts/...`)

## Obtaining FE-Schrift

FE-Schrift is the typeface mandated for Mercosul license plates. It is the same
font used on German license plates.

**Option A: Download from a free source**

```bash
# From the German government's official font repository
wget "https://github.com/aktion-hip/fe-schrift/raw/master/FE-Schrift.otf" \
     -O synthetic_plates/fonts/FE-Schrift.ttf
```

**Option B: Install via system package (Debian/Ubuntu)**

```bash
sudo apt install fonts-dejavu-core
```

**Option C: Download DejaVu Sans Mono Bold (best free fallback)**

```bash
wget "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSansMono-Bold.ttf" \
     -O synthetic_plates/fonts/DejaVuSansMono-Bold.ttf
```

## Verify detected fonts

```bash
python -m synthetic_plates fonts
```

Shows which font is being used (marked with ✓).

## Without any font

If no font is found, Pillow's default bitmap font is used. This works but
produces low-resolution text. For best results, install at least DejaVu Sans Mono Bold.